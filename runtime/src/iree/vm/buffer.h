// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BUFFER_H_
#define IREE_VM_BUFFER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Describes where a byte buffer originates from, what guarantees can be made
// about its lifetime and ownership, and how it may be accessed.
// Note that buffers may always be read.
enum iree_vm_buffer_access_bits_t {
  // The guest is allowed to write to the buffer.
  // If not specified the buffer is read-only.
  IREE_VM_BUFFER_ACCESS_MUTABLE = 1u << 0,

  // Buffer references memory in the module space (rodata or rwdata) that is
  // guaranteed to be live for the lifetime of the module.
  IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE = 1u << 1,
  // Buffer references memory created by the guest module code. It has a
  // lifetime less than that of the module but is always tracked with proper
  // references (a handle existing to the memory implies it is valid).
  IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST = 1u << 2,
  // Buffer references external host memory with an unknown lifetime.
  IREE_VM_BUFFER_ACCESS_ORIGIN_HOST = 1u << 3,
};
typedef uint32_t iree_vm_buffer_access_t;

// A simple byte range with options for ownership and wrapping semantics.
// The access flags indicate what access is allowed from the VM.
// Buffers are fixed-length and may only contain primitive values.
// For resizable lists with mixed element types and ref objects use
// iree_vm_list_t.
//
// Note that because buffers are just bags of bytes endianness issues are very
// likely depending on usage. In general IREE takes the stance that
// little-endian is all that is practically relevant nowadays and big-endian
// targets will need their own modules compiled with such a setting. This is to
// avoid the significant amount of work trying to ensure cross-endian
// correctness in things like packed .rodata, cross-device switching (host in
// a different endianness than HAL device), etc.
//
// For stack-allocated buffers setup with iree_vm_buffer_initialize the
// allocator provided will be used to free the data when the buffer is
// deinitialized. It may be iree_allocator_null to indicate the data is unowned.
//
// For heap-allocated buffers created with iree_vm_buffer_create/clone/etc the
// allocator is used to free the entire iree_vm_buffer_t and the co-allocated
// buffer data that lives after it in memory.
typedef struct iree_vm_buffer_t {
  iree_vm_ref_object_t ref_object;
  iree_vm_buffer_access_t access;
  iree_byte_span_t data;
  iree_allocator_t allocator;
} iree_vm_buffer_t;

// Initializes a buffer in-place with the given byte contents.
// This can be used to avoid buffer allocation overhead when wrapping existing
// buffers for API interop but buffer lifetime must be observed carefully by
// the caller.
//
// Some systems may assume that the data is aligned to at least the natural
// word size of the machine. If possible align to iree_max_align_t.
//
// |data| will be freed with |allocator| when the buffer is deinitialized.
// If the data is not owned then iree_allocator_null can be used to no-op the
// free.
//
// |access| can be used to control who (guest, host, etc) and how (read/write)
// the buffer may be accessed. If the allocation being wrapped has its own
// access requirements (read-only, etc) the caller must specify those flags.
IREE_API_EXPORT void iree_vm_buffer_initialize(iree_vm_buffer_access_t access,
                                               iree_byte_span_t data,
                                               iree_allocator_t allocator,
                                               iree_vm_buffer_t* out_buffer);

// Deinitializes a buffer previously initialized in-place with
// iree_vm_buffer_initialize. Invalid to call on a buffer that was allocated
// on the heap via iree_vm_buffer_create. Aborts if there are still references
// remaining.
IREE_API_EXPORT void iree_vm_buffer_deinitialize(iree_vm_buffer_t* buffer);

// Creates a new zero-initialized buffer of the given byte |length|.
// The underlying storage buffer may be allocated larger to ensure alignment.
// The allocated data will be aligned to iree_max_align_t.
//
// |access| can be used to control who (guest, host, etc) and how (read/write)
// the buffer may be accessed.
IREE_API_EXPORT iree_status_t iree_vm_buffer_create(
    iree_vm_buffer_access_t access, iree_host_size_t length,
    iree_allocator_t allocator, iree_vm_buffer_t** out_buffer);

// Retains the given |buffer| for the caller.
IREE_API_EXPORT void iree_vm_buffer_retain(iree_vm_buffer_t* buffer);

// Releases the given |buffer| from the caller.
IREE_API_EXPORT void iree_vm_buffer_release(iree_vm_buffer_t* buffer);

// Clones a range of bytes in |source| to a new buffer.
// The allocated data will be aligned to iree_max_align_t.
//
// |access| can be used to control who (guest, host, etc) and how (read/write)
// the buffer may be accessed. As this returns a newly allocated buffer the
// new access may be more permissive than the source buffer.
IREE_API_EXPORT iree_status_t iree_vm_buffer_clone(
    iree_vm_buffer_access_t access, const iree_vm_buffer_t* source_buffer,
    iree_host_size_t source_offset, iree_host_size_t length,
    iree_allocator_t allocator, iree_vm_buffer_t** out_buffer);

// Returns the user-visible length of the buffer in bytes.
IREE_API_EXPORT iree_host_size_t
iree_vm_buffer_length(const iree_vm_buffer_t* buffer);

// Returns the underlying data storage for the buffer.
// WARNING: this performs no validation of the access allowance on the buffer
// and the caller is responsible for all range checking. Use with caution and
// prefer the utility methods instead.
IREE_API_EXPORT uint8_t* iree_vm_buffer_data(const iree_vm_buffer_t* buffer);

// Copies a byte range from |source_buffer| to |target_buffer|.
IREE_API_EXPORT iree_status_t iree_vm_buffer_copy_bytes(
    const iree_vm_buffer_t* source_buffer, iree_host_size_t source_offset,
    const iree_vm_buffer_t* target_buffer, iree_host_size_t target_offset,
    iree_host_size_t length);

// Compares |lhs_buffer| to |rhs_buffer| for bitwise equality.
// |out_result| will receive 1 if the byte ranges are equal and 0 otherwise.
IREE_API_EXPORT iree_status_t iree_vm_buffer_compare_bytes(
    const iree_vm_buffer_t* lhs_buffer, iree_host_size_t lhs_offset,
    const iree_vm_buffer_t* rhs_buffer, iree_host_size_t rhs_offset,
    iree_host_size_t length, bool* out_result);

// Fills a byte range of |target_buffer| with the byte pattern.
IREE_API_EXPORT iree_status_t iree_vm_buffer_fill_bytes(
    const iree_vm_buffer_t* target_buffer, iree_host_size_t target_offset,
    iree_host_size_t length, uint8_t value);

// Fills an element range of |buffer| with the given pattern.
// Only |pattern_length| values with 1, 2, 4, or 8 bytes are supported.
// The |target_offset|, in bytes, must match the alignment of the pattern.
IREE_API_EXPORT iree_status_t iree_vm_buffer_fill_elements(
    const iree_vm_buffer_t* target_buffer, iree_host_size_t target_offset,
    iree_host_size_t element_count, iree_host_size_t element_length,
    const void* value);

// Maps a subrange to a span of bytes within the |buffer| for read-only access.
// |offset| and |length| must match the provided |alignment| (1, 2, 4, 8) and
// will be rounded toward zero if they do not.
IREE_API_EXPORT iree_status_t
iree_vm_buffer_map_ro(const iree_vm_buffer_t* buffer, iree_host_size_t offset,
                      iree_host_size_t length, iree_host_size_t alignment,
                      iree_const_byte_span_t* out_span);

// Maps a subrange to a span of bytes within the |buffer| for read/write access.
// |offset| and |length| must match the provided |alignment| (1, 2, 4, 8) and
// will be rounded toward zero if they do not.
IREE_API_EXPORT iree_status_t
iree_vm_buffer_map_rw(const iree_vm_buffer_t* buffer, iree_host_size_t offset,
                      iree_host_size_t length, iree_host_size_t alignment,
                      iree_byte_span_t* out_span);

// Reads |element_count| elements each of |element_length| bytes from the
// |source_buffer| into |out_target_ptr|. The |source_offset|, in bytes, must be
// aligned to at least the |element_length|.
// This routine performs checks on bounds, alignment, and access rights.
IREE_API_EXPORT iree_status_t iree_vm_buffer_read_elements(
    const iree_vm_buffer_t* source_buffer, iree_host_size_t source_offset,
    void* target_ptr, iree_host_size_t element_count,
    iree_host_size_t element_length);

// Writes |element_count| elements each of |element_length| bytes to the
// |target_buffer| from |source_ptr|. The |target_offset|, in bytes, must be
// aligned to at least the |element_length|.
// This routine performs checks on bounds, alignment, and access rights.
IREE_API_EXPORT iree_status_t iree_vm_buffer_write_elements(
    const void* source_ptr, const iree_vm_buffer_t* target_buffer,
    iree_host_size_t target_offset, iree_host_size_t element_count,
    iree_host_size_t element_length);

// Returns the a string view referencing the given |value| buffer.
// The returned view will only be valid for as long as the buffer is live.
static inline iree_string_view_t iree_vm_buffer_as_string(
    const iree_vm_buffer_t* value) {
  return value ? iree_make_string_view((const char*)value->data.data,
                                       value->data.data_length)
               : iree_string_view_empty();
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_vm_buffer, iree_vm_buffer_t);

#endif  // IREE_VM_BUFFER_H_
