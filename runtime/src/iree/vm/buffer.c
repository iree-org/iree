// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/buffer.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/vm/instance.h"

static iree_vm_ref_type_descriptor_t iree_vm_buffer_descriptor = {0};

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_vm_buffer, iree_vm_buffer_t);

static iree_status_t iree_vm_buffer_map(const iree_vm_buffer_t* buffer,
                                        iree_host_size_t offset,
                                        iree_host_size_t length,
                                        iree_host_size_t alignment,
                                        uint8_t** out_data,
                                        iree_host_size_t* out_data_length) {
  // Force alignment.
  offset &= ~(alignment - 1);
  length &= ~(alignment - 1);
  const iree_host_size_t end = offset + length;
  if (IREE_UNLIKELY(end > buffer->data.data_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "out-of-bounds access detected (offset=%zu, "
                            "length=%zu, alignment=%zu, buffer length=%zu)",
                            offset, length, alignment,
                            buffer->data.data_length);
  }
  *out_data = buffer->data.data + offset;
  *out_data_length = length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_vm_buffer_map_ro(const iree_vm_buffer_t* buffer, iree_host_size_t offset,
                      iree_host_size_t length, iree_host_size_t alignment,
                      iree_const_byte_span_t* out_span) {
  // Always allowed regardless of access.
  return iree_vm_buffer_map(buffer, offset, length, alignment,
                            (uint8_t**)&out_span->data, &out_span->data_length);
}

// Maps a subrange to a span of bytes within the |buffer| for read/write access.
// |offset| and |length| must match the provided |alignment| (1, 2, 4, 8) and
// will be rounded toward zero if they do not.
IREE_API_EXPORT iree_status_t
iree_vm_buffer_map_rw(const iree_vm_buffer_t* buffer, iree_host_size_t offset,
                      iree_host_size_t length, iree_host_size_t alignment,
                      iree_byte_span_t* out_span) {
  // Buffer requires mutable access.
  if (IREE_UNLIKELY(
          !iree_all_bits_set(buffer->access, IREE_VM_BUFFER_ACCESS_MUTABLE))) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "buffer is read-only and cannot be mapped for mutation");
  }
  return iree_vm_buffer_map(buffer, offset, length, alignment, &out_span->data,
                            &out_span->data_length);
}

IREE_API_EXPORT void iree_vm_buffer_initialize(iree_vm_buffer_access_t access,
                                               iree_byte_span_t data,
                                               iree_allocator_t allocator,
                                               iree_vm_buffer_t* out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  iree_atomic_ref_count_init(&out_buffer->ref_object.counter);
  out_buffer->access = access;
  out_buffer->data = data;
  out_buffer->allocator = allocator;
}

IREE_API_EXPORT void iree_vm_buffer_deinitialize(iree_vm_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  iree_atomic_ref_count_abort_if_uses(&buffer->ref_object.counter);
  iree_allocator_free(buffer->allocator, buffer->data.data);
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_create(
    iree_vm_buffer_access_t access, iree_host_size_t length,
    iree_allocator_t allocator, iree_vm_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // The actual buffer payload is prefixed with the buffer type so we need only
  // a single allocation.
  iree_host_size_t prefix_size = iree_sizeof_struct(**out_buffer);
  iree_host_size_t total_size = prefix_size + length;

  // Allocate combined [prefix | buffer] memory.
  uint8_t* data_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&data_ptr));

  // Initialize the prefix buffer handle.
  iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)data_ptr;
  memset(data_ptr, 0, prefix_size - sizeof(*buffer));  // padding
  iree_byte_span_t target_span =
      iree_make_byte_span(data_ptr + prefix_size, length);
  iree_vm_buffer_initialize(access, target_span, allocator, buffer);

  *out_buffer = buffer;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_vm_buffer_destroy(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Buffers are stored as [prefix | data]; freeing the prefix is all we need
  // to do to free it all.
  iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)ptr;
  iree_allocator_free(buffer->allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_vm_buffer_retain(iree_vm_buffer_t* buffer) {
  iree_vm_ref_object_retain(buffer, &iree_vm_buffer_descriptor);
}

IREE_API_EXPORT void iree_vm_buffer_release(iree_vm_buffer_t* buffer) {
  iree_vm_ref_object_release(buffer, &iree_vm_buffer_descriptor);
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_clone(
    iree_vm_buffer_access_t access, const iree_vm_buffer_t* source_buffer,
    iree_host_size_t source_offset, iree_host_size_t length,
    iree_allocator_t allocator, iree_vm_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Try to map the source buffer first; no use continuing if we can't read the
  // data to clone.
  iree_const_byte_span_t source_span;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_buffer_map_ro(source_buffer, source_offset, length, 1,
                                &source_span));

  // The actual buffer payload is prefixed with the buffer type so we need only
  // a single allocation.
  iree_host_size_t prefix_size =
      iree_host_align(sizeof(iree_vm_buffer_t), iree_max_align_t);
  iree_host_size_t total_size = prefix_size + source_span.data_length;

  // Allocate combined [prefix | buffer] memory.
  // NOTE: we are allocating without initialization here as we will be writing
  // over all of it.
  uint8_t* data_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_uninitialized(allocator, total_size,
                                              (void**)&data_ptr));

  // Initialize the prefix buffer handle.
  iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)data_ptr;
  memset(data_ptr, 0, prefix_size - sizeof(*buffer));  // padding
  iree_byte_span_t target_span =
      iree_make_byte_span(data_ptr + prefix_size, length);
  iree_vm_buffer_initialize(access, target_span, allocator, buffer);

  // Copy the data from the source buffer.
  memcpy(target_span.data, source_span.data, target_span.data_length);

  *out_buffer = buffer;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_host_size_t
iree_vm_buffer_length(const iree_vm_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->data.data_length;
}

IREE_API_EXPORT uint8_t* iree_vm_buffer_data(const iree_vm_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->data.data;
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_copy_bytes(
    const iree_vm_buffer_t* source_buffer, iree_host_size_t source_offset,
    const iree_vm_buffer_t* target_buffer, iree_host_size_t target_offset,
    iree_host_size_t length) {
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  iree_const_byte_span_t source_span;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(source_buffer, source_offset,
                                             length, 1, &source_span));
  iree_byte_span_t target_span;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_rw(target_buffer, target_offset,
                                             length, 1, &target_span));
  memcpy(target_span.data, source_span.data, length);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_compare_bytes(
    const iree_vm_buffer_t* lhs_buffer, iree_host_size_t lhs_offset,
    const iree_vm_buffer_t* rhs_buffer, iree_host_size_t rhs_offset,
    iree_host_size_t length, bool* out_result) {
  IREE_ASSERT_ARGUMENT(lhs_buffer);
  IREE_ASSERT_ARGUMENT(rhs_buffer);
  iree_const_byte_span_t lhs_span;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_map_ro(lhs_buffer, lhs_offset, length, 1, &lhs_span));
  iree_const_byte_span_t rhs_span;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_map_ro(rhs_buffer, rhs_offset, length, 1, &rhs_span));
  *out_result = memcmp(lhs_span.data, rhs_span.data, length) == 0;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_fill_bytes(
    const iree_vm_buffer_t* target_buffer, iree_host_size_t target_offset,
    iree_host_size_t length, uint8_t value) {
  return iree_vm_buffer_fill_elements(target_buffer, target_offset, length, 1,
                                      &value);
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_fill_elements(
    const iree_vm_buffer_t* target_buffer, iree_host_size_t target_offset,
    iree_host_size_t element_count, iree_host_size_t element_length,
    const void* value) {
  IREE_ASSERT_ARGUMENT(target_buffer);
  iree_byte_span_t span;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_rw(target_buffer, target_offset,
                                             element_count * element_length,
                                             element_length, &span));
  switch (element_length) {
    case 1: {
      const uint8_t pattern_value = *(const uint8_t*)value;
      memset(span.data, pattern_value, span.data_length);
    } break;
    case 2: {
      const uint16_t pattern_value = *(const uint16_t*)value;
      uint16_t* target_ptr = (uint16_t*)span.data;
      for (iree_host_size_t i = 0; i < element_count; ++i) {
        target_ptr[i] = pattern_value;
      }
    } break;
    case 4: {
      const uint32_t pattern_value = *(const uint32_t*)value;
      uint32_t* target_ptr = (uint32_t*)span.data;
      for (iree_host_size_t i = 0; i < element_count; ++i) {
        target_ptr[i] = pattern_value;
      }
    } break;
    case 8: {
      const uint64_t pattern_value = *(const uint64_t*)value;
      uint64_t* target_ptr = (uint64_t*)span.data;
      for (iree_host_size_t i = 0; i < element_count; ++i) {
        target_ptr[i] = pattern_value;
      }
    } break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid element length %" PRIhsz
                              "; expected one of [1, 2, 4, 8]",
                              element_length);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_read_elements(
    const iree_vm_buffer_t* source_buffer, iree_host_size_t source_offset,
    void* target_ptr, iree_host_size_t element_count,
    iree_host_size_t element_length) {
  IREE_ASSERT_ARGUMENT(source_buffer);
  iree_const_byte_span_t source_span;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(source_buffer, source_offset,
                                             element_count * element_length,
                                             element_length, &source_span));
  memcpy(target_ptr, source_span.data, source_span.data_length);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_write_elements(
    const void* source_ptr, const iree_vm_buffer_t* target_buffer,
    iree_host_size_t target_offset, iree_host_size_t element_count,
    iree_host_size_t element_length) {
  IREE_ASSERT_ARGUMENT(source_ptr);
  IREE_ASSERT_ARGUMENT(target_buffer);
  iree_byte_span_t target_span;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_rw(target_buffer, target_offset,
                                             element_count * element_length,
                                             element_length, &target_span));
  memcpy(target_span.data, source_ptr, target_span.data_length);
  return iree_ok_status();
}

iree_status_t iree_vm_buffer_register_types(iree_vm_instance_t* instance) {
  if (iree_vm_buffer_descriptor.type != IREE_VM_REF_TYPE_NULL) {
    // Already registered.
    return iree_ok_status();
  }

  iree_vm_buffer_descriptor.destroy = iree_vm_buffer_destroy;
  iree_vm_buffer_descriptor.offsetof_counter =
      offsetof(iree_vm_buffer_t, ref_object.counter);
  iree_vm_buffer_descriptor.type_name = iree_make_cstring_view("vm.buffer");
  return iree_vm_ref_register_type(&iree_vm_buffer_descriptor);
}
