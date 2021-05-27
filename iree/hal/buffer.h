// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_H_
#define IREE_HAL_BUFFER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_allocator_s iree_hal_allocator_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Whole length of the underlying buffer.
#define IREE_WHOLE_BUFFER ((iree_device_size_t)(-1))

// A bitfield specifying properties for a memory type.
enum iree_hal_memory_type_e {
  IREE_HAL_MEMORY_TYPE_NONE = 0u,

  // Memory is lazily allocated by the device and only exists transiently.
  // This is the optimal mode for memory used only within a single command
  // buffer. Transient buffers, even if they have
  // IREE_HAL_MEMORY_TYPE_HOST_VISIBLE set, should be treated as device-local
  // and opaque as they may have no memory attached to them outside of the time
  // they are being evaluated on devices.
  //
  // This flag can be treated as a hint in most cases; allocating a buffer with
  // it set _may_ return the same as if it had not be set. Certain allocation
  // routines may use the hint to more tightly control reuse or defer wiring the
  // memory.
  IREE_HAL_MEMORY_TYPE_TRANSIENT = 1u << 0,

  // Memory allocated with this type can be mapped for host access using
  // iree_hal_buffer_map_range.
  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE = 1u << 1,

  // The host cache management commands MappedMemory::Flush and
  // MappedMemory::Invalidate are not needed to flush host writes
  // to the device or make device writes visible to the host, respectively.
  IREE_HAL_MEMORY_TYPE_HOST_COHERENT = 1u << 2,

  // Memory allocated with this type is cached on the host. Host memory
  // accesses to uncached memory are slower than to cached memory, however
  // uncached memory is always host coherent. MappedMemory::Flush must be used
  // to ensure the device has visibility into any changes made on the host and
  // Invalidate must be used to ensure the host has visibility into any changes
  // made on the device.
  IREE_HAL_MEMORY_TYPE_HOST_CACHED = 1u << 3,

  // Memory is accessible as normal host allocated memory.
  IREE_HAL_MEMORY_TYPE_HOST_LOCAL =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,

  // Memory allocated with this type is visible to the device for execution.
  // Being device visible does not mean the same thing as
  // IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL. Though an allocation may be visible to
  // the device and therefore useable for execution it may require expensive
  // mapping or implicit transfers.
  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE = 1u << 4,

  // Memory allocated with this type is the most efficient for device access.
  // Devices may support using memory that is not device local via
  // IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE but doing so can incur non-trivial
  // performance penalties. Device local memory, on the other hand, is
  // guaranteed to be fast for all operations.
  IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL =
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE | (1u << 5),
};
typedef uint32_t iree_hal_memory_type_t;

// A bitfield specifying how memory will be accessed in a mapped memory region.
enum iree_hal_memory_access_e {
  // Memory is not mapped.
  IREE_HAL_MEMORY_ACCESS_NONE = 0u,
  // Memory will be read.
  // If a buffer is only mapped for reading it may still be possible to write to
  // it but the results will be undefined (as it may present coherency issues).
  IREE_HAL_MEMORY_ACCESS_READ = 1u << 0,
  // Memory will be written.
  // If a buffer is only mapped for writing it may still be possible to read
  // from it but the results will be undefined or incredibly slow (as it may
  // be mapped by the driver as uncached).
  IREE_HAL_MEMORY_ACCESS_WRITE = 1u << 1,
  // Memory will be discarded prior to mapping.
  // The existing contents will be undefined after mapping and must be written
  // to ensure validity.
  IREE_HAL_MEMORY_ACCESS_DISCARD = 1u << 2,
  // Memory will be discarded and completely overwritten in a single operation.
  IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE =
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_DISCARD,
  // A flag that can be applied to any access type to indicate that the buffer
  // storage being accessed may alias with other accesses occurring concurrently
  // within or across operations. The lack of the flag indicates that the access
  // is guaranteed not to alias (ala C's `restrict` keyword).
  IREE_HAL_MEMORY_ACCESS_MAY_ALIAS = 1u << 3,
  // Memory may have any operation performed on it.
  IREE_HAL_MEMORY_ACCESS_ALL = IREE_HAL_MEMORY_ACCESS_READ |
                               IREE_HAL_MEMORY_ACCESS_WRITE |
                               IREE_HAL_MEMORY_ACCESS_DISCARD,
};
typedef uint32_t iree_hal_memory_access_t;

// Bitfield that defines how a buffer is intended to be used.
// Usage allows the driver to appropriately place the buffer for more
// efficient operations of the specified types.
enum iree_hal_buffer_usage_e {
  IREE_HAL_BUFFER_USAGE_NONE = 0u,

  // The buffer, once defined, will not be mapped or updated again.
  // This should be used for uniform parameter values such as runtime
  // constants for executables. Doing so may allow drivers to inline values or
  // represent them in command buffers more efficiently (avoiding memory reads
  // or swapping, etc).
  IREE_HAL_BUFFER_USAGE_CONSTANT = 1u << 0,

  // The buffer can be used as the source or target of a transfer command
  // (CopyBuffer, UpdateBuffer, etc).
  //
  // If |IREE_HAL_BUFFER_USAGE_MAPPING| is not specified drivers may safely
  // assume that the host may never need visibility of this buffer as all
  // accesses will happen via command buffers.
  IREE_HAL_BUFFER_USAGE_TRANSFER = 1u << 1,

  // The buffer can be mapped by the host application for reading and writing.
  //
  // As mapping may require placement in special address ranges or system
  // calls to enable visibility the driver can use the presence (or lack of)
  // this flag to perform allocation-type setup and avoid initial mapping
  // overhead.
  IREE_HAL_BUFFER_USAGE_MAPPING = 1u << 2,

  // The buffer can be provided as an input or output to an executable.
  // Buffers of this type may be directly used by drivers during dispatch.
  IREE_HAL_BUFFER_USAGE_DISPATCH = 1u << 3,

  // Buffer may be used for any operation.
  IREE_HAL_BUFFER_USAGE_ALL = IREE_HAL_BUFFER_USAGE_TRANSFER |
                              IREE_HAL_BUFFER_USAGE_MAPPING |
                              IREE_HAL_BUFFER_USAGE_DISPATCH,
};
typedef uint32_t iree_hal_buffer_usage_t;

// Buffer overlap testing results.
enum iree_hal_buffer_overlap_e {
  // No overlap between the two buffers.
  IREE_HAL_BUFFER_OVERLAP_DISJOINT = 0,
  // Partial overlap between the two buffers.
  IREE_HAL_BUFFER_OVERLAP_PARTIAL,
  // Complete overlap between the two buffers (they are the same).
  IREE_HAL_BUFFER_OVERLAP_COMPLETE,
};
typedef uint8_t iree_hal_buffer_overlap_t;

enum iree_hal_mapping_mode_e {
  IREE_HAL_MAPPING_MODE_SCOPED = 0,
  IREE_HAL_MAPPING_MODE_PERSISTENT = 0,
};
typedef uint32_t iree_hal_mapping_mode_t;

// Reference to a buffer's mapped memory.
typedef struct {
  // Contents of the buffer. Behavior is undefined if an access is performed
  // whose type was not specified during mapping.
  //
  // The bytes available may be greater than what was requested if platform
  // alignment rules require it. Only memory defined by the given span may be
  // accessed.
  iree_byte_span_t contents;

  // Used internally - do not modify.
  uint64_t reserved[4];
} iree_hal_buffer_mapping_t;

// TODO(benvanik): replace with tables for iree_string_builder_*.
#define iree_hal_memory_type_string(...) "TODO"
//     // Combined:
//     {IREE_HAL_MEMORY_TYPE_HOST_LOCAL, "HOST_LOCAL"},
//     {IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, "DEVICE_LOCAL"},
//     // Separate:
//     {IREE_HAL_MEMORY_TYPE_TRANSIENT, "TRANSIENT"},
//     {IREE_HAL_MEMORY_TYPE_HOST_VISIBLE, "HOST_VISIBLE"},
//     {IREE_HAL_MEMORY_TYPE_HOST_COHERENT, "HOST_COHERENT"},
//     {IREE_HAL_MEMORY_TYPE_HOST_CACHED, "HOST_CACHED"},
//     {IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, "DEVICE_VISIBLE"},
#define iree_hal_memory_access_string(...) "TODO"
//     // Combined:
//     {IREE_HAL_MEMORY_ACCESS_ALL, "ALL"},
//     {IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, "DISCARD_WRITE"},
//     // Separate:
//     {IREE_HAL_MEMORY_ACCESS_READ, "READ"},
//     {IREE_HAL_MEMORY_ACCESS_WRITE, "WRITE"},
//     {IREE_HAL_MEMORY_ACCESS_DISCARD, "DISCARD"},
//     {IREE_HAL_MEMORY_ACCESS_MAY_ALIAS, "MAY_ALIAS"},
#define iree_hal_buffer_usage_string(...) "TODO"
//     // Combined:
//     {IREE_HAL_BUFFER_USAGE_ALL, "ALL"},
//     // Separate:
//     {IREE_HAL_BUFFER_USAGE_CONSTANT, "CONSTANT"},
//     {IREE_HAL_BUFFER_USAGE_TRANSFER, "TRANSFER"},
//     {IREE_HAL_BUFFER_USAGE_MAPPING, "MAPPING"},
//     {IREE_HAL_BUFFER_USAGE_DISPATCH, "DISPATCH"},

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

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
// The memory types (defined by a bitfield of iree_hal_memory_type_t values)
// that a particular context (host or device) may use vary from device to device
// and must be queried by the application when allocating buffers. It's strongly
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
// a device. iree_hal_buffer_Subspan can be used to reference subspans of
// buffers like absl::Span - though unlike absl::Span the returned Buffer holds
// a reference to the parent buffer.
typedef struct iree_hal_buffer_s iree_hal_buffer_t;

// Returns success iff the buffer was allocated with the given memory type.
IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_memory_type(
    iree_hal_memory_type_t actual_memory_type,
    iree_hal_memory_type_t expected_memory_type);

// Returns success iff the buffer allows the requested access.
IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_access(
    iree_hal_memory_access_t allowed_memory_access,
    iree_hal_memory_access_t required_memory_access);

// Returns success iff the buffer usage allows the given usage type.
IREE_API_EXPORT iree_status_t
iree_hal_buffer_validate_usage(iree_hal_buffer_usage_t allowed_usage,
                               iree_hal_buffer_usage_t required_usage);

// Returns success iff the given byte range falls within the valid buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length);

// Tests whether the given buffers overlap, including support for subspans.
// IREE_WHOLE_BUFFER may be used for |lhs_length| and/or |rhs_length| to use the
// lengths of those buffers, respectively.
IREE_API_EXPORT iree_hal_buffer_overlap_t iree_hal_buffer_test_overlap(
    iree_hal_buffer_t* lhs_buffer, iree_device_size_t lhs_offset,
    iree_device_size_t lhs_length, iree_hal_buffer_t* rhs_buffer,
    iree_device_size_t rhs_offset, iree_device_size_t rhs_length);

// Returns a reference to a subspan of the |buffer|.
// If |byte_length| is IREE_WHOLE_BUFFER the remaining bytes in the buffer after
// |byte_offset| (possibly 0) will be selected.
//
// The parent buffer will remain alive for the lifetime of the subspan
// returned. If the subspan is a small portion this may cause additional
// memory to remain allocated longer than required.
//
// Returns the given |buffer| if the requested span covers the entire range.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_hal_buffer_subspan(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer);

// Retains the given |buffer| for the caller.
IREE_API_EXPORT void iree_hal_buffer_retain(iree_hal_buffer_t* buffer);

// Releases the given |buffer| from the caller.
IREE_API_EXPORT void iree_hal_buffer_release(iree_hal_buffer_t* buffer);

// Returns the allocator this buffer was allocated from.
IREE_API_EXPORT iree_hal_allocator_t* iree_hal_buffer_allocator(
    const iree_hal_buffer_t* buffer);

// Returns a pointer to the buffer containing the actual allocation.
// The buffer represents a span of the allocated bytes defined by byte_offset
// and byte_length. If the provided buffer *is* the allocated buffer then the
// returned value will be the provided buffer pointer.
IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_allocated_buffer(
    const iree_hal_buffer_t* buffer);

// Returns the size of the resource memory allocation in bytes.
// This may be rounded up from the originally requested size or the ideal
// size for the resource based on device restrictions.
IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_allocation_size(const iree_hal_buffer_t* buffer);

// Returns the offset in bytes of the buffer within its allocated_buffer.
IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_offset(const iree_hal_buffer_t* buffer);

// Returns the size in bytes of the buffer.
IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_length(const iree_hal_buffer_t* buffer);

// Returns the memory type the buffer was allocated with.
IREE_API_EXPORT
iree_hal_memory_type_t iree_hal_buffer_memory_type(
    const iree_hal_buffer_t* buffer);

// Returns the allowed memory access modes.
// These may be more strict than the underlying allocation, for example when the
// buffer is exposing read-only memory that may be in mutable pages.
IREE_API_EXPORT
iree_hal_memory_access_t iree_hal_buffer_allowed_access(
    const iree_hal_buffer_t* buffer);

// Returns the allowed buffer usage modes.
IREE_API_EXPORT
iree_hal_buffer_usage_t iree_hal_buffer_allowed_usage(
    const iree_hal_buffer_t* buffer);

// Sets a range of the buffer to binary zero.
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// The byte range in |buffer| will be flushed if needed.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_fill_buffer.
IREE_API_EXPORT iree_status_t
iree_hal_buffer_zero(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length);

// Sets a range of the buffer to the given value.
// Only |pattern_length| values with 1, 2, or 4 bytes are supported.
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// The byte range in |buffer| will be flushed if needed.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_fill_buffer.
IREE_API_EXPORT iree_status_t
iree_hal_buffer_fill(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length, const void* pattern,
                     iree_host_size_t pattern_length);

// Reads a block of data from the buffer at the given offset.
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_copy_buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_read_data(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length);

// Writes a block of byte data into the buffer at the given offset.
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// The byte range in |target_buffer| will be flushed if needed.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_update_buffer and
// iree_hal_command_buffer_copy_buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_write_data(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length);

// Copies data from the provided |source_buffer| into the |target_buffer|.
//
// Requires that both buffers have the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// The byte range in |target_buffer| will be flushed if needed.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_copy_buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_copy_data(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t data_length);

// Maps the buffer to be accessed as a host pointer into |out_buffer_mapping|.
// The byte offset and byte length may be adjusted for device alignment.
// The output data pointer will be properly aligned to the start of the data.
// Fails if the memory could not be mapped (invalid access type, invalid
// range, or unsupported memory type).
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// If the buffer is not IREE_HAL_MEMORY_TYPE_HOST_COHERENT then the caller must
// invalidate the byte range they want to access to update the visibility of the
// mapped memory.
IREE_API_EXPORT iree_status_t iree_hal_buffer_map_range(
    iree_hal_buffer_t* buffer, iree_hal_memory_access_t memory_access,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_mapping_t* out_buffer_mapping);

// Unmaps the buffer as was previously mapped to |buffer_mapping|.
//
// If the buffer is not IREE_HAL_MEMORY_TYPE_HOST_COHERENT then the caller must
// flush the byte range they want to make available to other threads/devices.
IREE_API_EXPORT void iree_hal_buffer_unmap_range(
    iree_hal_buffer_mapping_t* buffer_mapping);

// Invalidates ranges of non-coherent memory from the host caches.
// This guarantees that device writes to the memory ranges provided are
// visible on the host. Use before reading from non-coherent memory.
//
// Only required for memory types without IREE_HAL_MEMORY_TYPE_HOST_COHERENT.
IREE_API_EXPORT iree_status_t iree_hal_buffer_invalidate_range(
    iree_hal_buffer_mapping_t* buffer_mapping, iree_device_size_t byte_offset,
    iree_device_size_t byte_length);

// Flushes ranges of non-coherent memory from the host caches.
// This guarantees that host writes to the memory ranges provided are available
// for device access. Use after writing to non-coherent memory.
//
// Only required for memory types without IREE_HAL_MEMORY_TYPE_HOST_COHERENT.
IREE_API_EXPORT iree_status_t iree_hal_buffer_flush_range(
    iree_hal_buffer_mapping_t* buffer_mapping, iree_device_size_t byte_offset,
    iree_device_size_t byte_length);

// Calculates and returns a byte subspan range within a buffer mapping.
// The byte range provided is local to the mapping. May return a 0-length span.
// IREE_WHOLE_BUFFER can be used for |byte_length|.
//
// Note that the access requirements of the mapping still hold: if the memory is
// not host coherent and writeable then the caller must use the
// iree_hal_buffer_invalidate_range and iree_hal_buffer_flush_range methods to
// ensure memory is in the expected state.
IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_subspan(
    iree_hal_buffer_mapping_t* buffer_mapping,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_byte_span_t* out_span);

//===----------------------------------------------------------------------===//
// iree_hal_heap_buffer_t
//===----------------------------------------------------------------------===//

// Wraps an existing host allocation in a buffer.
// When the buffer is destroyed the provided |data_allocator| will be used to
// free |data|. Pass iree_allocator_null() to wrap without ownership semantics.
//
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_hal_heap_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_byte_span_t data, iree_allocator_t data_allocator,
    iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t implementation details
//===----------------------------------------------------------------------===//

typedef struct {
  // << HAL C porting in progress >>
  IREE_API_UNSTABLE

  void(IREE_API_PTR* destroy)(iree_hal_buffer_t* buffer);

  iree_status_t(IREE_API_PTR* map_range)(iree_hal_buffer_t* buffer,
                                         iree_hal_mapping_mode_t mapping_mode,
                                         iree_hal_memory_access_t memory_access,
                                         iree_device_size_t local_byte_offset,
                                         iree_device_size_t local_byte_length,
                                         void** out_data_ptr);

  void(IREE_API_PTR* unmap_range)(iree_hal_buffer_t* buffer,
                                  iree_device_size_t local_byte_offset,
                                  iree_device_size_t local_byte_length,
                                  void* data_ptr);

  iree_status_t(IREE_API_PTR* invalidate_range)(
      iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length);

  iree_status_t(IREE_API_PTR* flush_range)(
      iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length);
} iree_hal_buffer_vtable_t;

struct iree_hal_buffer_s {
  iree_hal_resource_t resource;

  iree_hal_allocator_t* allocator;

  iree_hal_buffer_t* allocated_buffer;
  iree_device_size_t allocation_size;
  iree_device_size_t byte_offset;
  iree_device_size_t byte_length;

  iree_hal_memory_type_t memory_type;
  iree_hal_memory_access_t allowed_access;
  iree_hal_buffer_usage_t allowed_usage;
};

IREE_API_EXPORT void iree_hal_buffer_destroy(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_H_
