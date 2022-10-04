// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/buffer.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/detail.h"

#define _VTABLE_DISPATCH(buffer, method_name) \
  IREE_HAL_VTABLE_DISPATCH(buffer, iree_hal_buffer, method_name)

//===----------------------------------------------------------------------===//
// String utils
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_string_view_t iree_hal_memory_type_format(
    iree_hal_memory_type_t value, iree_bitfield_string_temp_t* out_temp) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      // Combined:
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_SVL("HOST_LOCAL")},
      {IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, IREE_SVL("DEVICE_LOCAL")},
      // Separate:
      {IREE_HAL_MEMORY_TYPE_OPTIMAL, IREE_SVL("OPTIMAL")},
      {IREE_HAL_MEMORY_TYPE_HOST_VISIBLE, IREE_SVL("HOST_VISIBLE")},
      {IREE_HAL_MEMORY_TYPE_HOST_COHERENT, IREE_SVL("HOST_COHERENT")},
      {IREE_HAL_MEMORY_TYPE_HOST_CACHED, IREE_SVL("HOST_CACHED")},
      {IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_SVL("DEVICE_VISIBLE")},
  };
  return iree_bitfield_format_inline(value, IREE_ARRAYSIZE(mappings), mappings,
                                     out_temp);
}

IREE_API_EXPORT iree_string_view_t iree_hal_memory_access_format(
    iree_hal_memory_access_t value, iree_bitfield_string_temp_t* out_temp) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      // Combined:
      {IREE_HAL_MEMORY_ACCESS_ALL, IREE_SVL("ALL")},
      {IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, IREE_SVL("DISCARD_WRITE")},
      // Separate:
      {IREE_HAL_MEMORY_ACCESS_READ, IREE_SVL("READ")},
      {IREE_HAL_MEMORY_ACCESS_WRITE, IREE_SVL("WRITE")},
      {IREE_HAL_MEMORY_ACCESS_DISCARD, IREE_SVL("DISCARD")},
      {IREE_HAL_MEMORY_ACCESS_MAY_ALIAS, IREE_SVL("MAY_ALIAS")},
      {IREE_HAL_MEMORY_ACCESS_ANY, IREE_SVL("ANY")},
  };
  return iree_bitfield_format_inline(value, IREE_ARRAYSIZE(mappings), mappings,
                                     out_temp);
}

IREE_API_EXPORT iree_string_view_t iree_hal_buffer_usage_format(
    iree_hal_buffer_usage_t value, iree_bitfield_string_temp_t* out_temp) {
  // clang-format off
  static const iree_bitfield_string_mapping_t mappings[] = {
    // Combined:
    {IREE_HAL_BUFFER_USAGE_TRANSFER, IREE_SVL("TRANSFER")},
    {IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE, IREE_SVL("DISPATCH_STORAGE")},
    {IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE, IREE_SVL("DISPATCH_IMAGE")},
    {IREE_HAL_BUFFER_USAGE_MAPPING, IREE_SVL("MAPPING")},
    // Separate:
    {IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE, IREE_SVL("TRANSFER_SOURCE")},
    {IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET, IREE_SVL("TRANSFER_TARGET")},
    {IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMS, IREE_SVL("DISPATCH_INDIRECT_PARAMS")},
    {IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ, IREE_SVL("DISPATCH_UNIFORM_READ")},
    {IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ, IREE_SVL("DISPATCH_STORAGE_READ")},
    {IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE, IREE_SVL("DISPATCH_STORAGE_WRITE")},
    {IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_READ, IREE_SVL("DISPATCH_IMAGE_READ")},
    {IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_WRITE, IREE_SVL("DISPATCH_IMAGE_WRITE")},
    {IREE_HAL_BUFFER_USAGE_SHARING_EXPORT, IREE_SVL("SHARING_EXPORT")},
    {IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE, IREE_SVL("SHARING_REPLICATE")},
    {IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT, IREE_SVL("SHARING_CONCURRENT")},
    {IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE, IREE_SVL("SHARING_IMMUTABLE")},
    {IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED, IREE_SVL("MAPPING_SCOPED")},
    {IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT, IREE_SVL("MAPPING_PERSISTENT")},
    {IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL, IREE_SVL("MAPPING_OPTIONAL")},
    {IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM, IREE_SVL("MAPPING_ACCESS_RANDOM")},
    {IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE, IREE_SVL("MAPPING_ACCESS_SEQUENTIAL_WRITE")},
  };
  return iree_bitfield_format_inline(value, IREE_ARRAYSIZE(mappings), mappings,
                                     out_temp);
}

//===----------------------------------------------------------------------===//
// Subspan indirection buffer
//===----------------------------------------------------------------------===//

static const iree_hal_buffer_vtable_t iree_hal_subspan_buffer_vtable;

IREE_API_EXPORT void iree_hal_subspan_buffer_initialize(
    iree_hal_buffer_t* allocated_buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_buffer_t* out_buffer) {
  IREE_ASSERT_ARGUMENT(allocated_buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  iree_hal_buffer_initialize(host_allocator, device_allocator, allocated_buffer,
                             allocated_buffer->allocation_size, byte_offset,
                             byte_length, allocated_buffer->memory_type,
                             allocated_buffer->allowed_access,
                             allocated_buffer->allowed_usage,
                             &iree_hal_subspan_buffer_vtable, out_buffer);
}

IREE_API_EXPORT void iree_hal_subspan_buffer_deinitialize(
    iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  iree_hal_buffer_release(buffer->allocated_buffer);
  buffer->allocated_buffer = NULL;
}

IREE_API_EXPORT iree_status_t iree_hal_subspan_buffer_create(
    iree_hal_buffer_t* allocated_buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocated_buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(
        host_allocator, device_allocator, allocated_buffer,
        allocated_buffer->allocation_size, byte_offset, byte_length,
        allocated_buffer->memory_type, allocated_buffer->allowed_access,
        allocated_buffer->allowed_usage, &iree_hal_subspan_buffer_vtable,
        buffer);
    *out_buffer = buffer;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_subspan_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_release(base_buffer->allocated_buffer);
  iree_allocator_free(host_allocator, base_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_subspan_buffer_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  return _VTABLE_DISPATCH(buffer->allocated_buffer, map_range)(
      buffer->allocated_buffer, mapping_mode, memory_access, local_byte_offset,
      local_byte_length, mapping);
}

static iree_status_t iree_hal_subspan_buffer_unmap_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  if (!buffer->allocated_buffer) return iree_ok_status();
  return _VTABLE_DISPATCH(buffer->allocated_buffer, unmap_range)(
      buffer->allocated_buffer, local_byte_offset, local_byte_length, mapping);
}

static iree_status_t iree_hal_subspan_buffer_invalidate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return _VTABLE_DISPATCH(buffer->allocated_buffer, invalidate_range)(
      buffer->allocated_buffer, local_byte_offset, local_byte_length);
}

static iree_status_t iree_hal_subspan_buffer_flush_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return _VTABLE_DISPATCH(buffer->allocated_buffer, flush_range)(
      buffer->allocated_buffer, local_byte_offset, local_byte_length);
}

static const iree_hal_buffer_vtable_t iree_hal_subspan_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_subspan_buffer_destroy,
    .map_range = iree_hal_subspan_buffer_map_range,
    .unmap_range = iree_hal_subspan_buffer_unmap_range,
    .invalidate_range = iree_hal_subspan_buffer_invalidate_range,
    .flush_range = iree_hal_subspan_buffer_flush_range,
};

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_buffer_initialize(
    iree_allocator_t host_allocator, iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_t* allocated_buffer, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    const iree_hal_buffer_vtable_t* vtable, iree_hal_buffer_t* buffer) {
  iree_hal_resource_initialize(vtable, &buffer->resource);
  buffer->host_allocator = host_allocator;
  buffer->device_allocator = device_allocator;
  buffer->allocated_buffer = allocated_buffer;
  buffer->allocation_size = allocation_size;
  buffer->byte_offset = byte_offset;
  buffer->byte_length = byte_length;
  buffer->memory_type = memory_type;
  buffer->allowed_access = allowed_access;
  buffer->allowed_usage = allowed_usage;

  // Retain the base allocated buffer if it's unique from the buffer we are
  // initializing.
  if (allocated_buffer != buffer) {
    iree_hal_buffer_retain(buffer->allocated_buffer);
  }
}

IREE_API_EXPORT void iree_hal_buffer_recycle(iree_hal_buffer_t* buffer) {
  if (IREE_LIKELY(buffer)) {
    IREE_TRACE_ZONE_BEGIN(z0);
    if (buffer->device_allocator) {
      iree_hal_allocator_deallocate_buffer(buffer->device_allocator, buffer);
    } else {
      iree_hal_buffer_destroy(buffer);
    }
    IREE_TRACE_ZONE_END(z0);
  }
}

IREE_API_EXPORT void iree_hal_buffer_destroy(iree_hal_buffer_t* buffer) {
  if (IREE_LIKELY(buffer)) {
    IREE_HAL_VTABLE_DISPATCH(buffer, iree_hal_buffer, destroy)
    (buffer);
  }
}

IREE_API_EXPORT void iree_hal_buffer_retain(iree_hal_buffer_t* buffer) {
  if (IREE_LIKELY(buffer)) {
    iree_atomic_ref_count_inc(&((iree_hal_resource_t*)(buffer))->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_buffer_release(iree_hal_buffer_t* buffer) {
  if (IREE_LIKELY(buffer) &&
      iree_atomic_ref_count_dec(&((iree_hal_resource_t*)(buffer))->ref_count) ==
          1) {
    iree_hal_buffer_recycle(buffer);
  }
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_memory_type(
    iree_hal_memory_type_t actual_memory_type,
    iree_hal_memory_type_t expected_memory_type) {
  if (IREE_UNLIKELY(
          !iree_all_bits_set(actual_memory_type, expected_memory_type))) {
#if IREE_STATUS_MODE
    // Missing one or more bits.
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t actual_memory_type_str =
        iree_hal_memory_type_format(actual_memory_type, &temp0);
    iree_string_view_t expected_memory_type_str =
        iree_hal_memory_type_format(expected_memory_type, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "buffer memory type is not compatible with the requested operation; "
        "buffer has %.*s, operation requires %.*s",
        (int)actual_memory_type_str.size, actual_memory_type_str.data,
        (int)expected_memory_type_str.size, expected_memory_type_str.data);
#else
    return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_access(
    iree_hal_memory_access_t allowed_memory_access,
    iree_hal_memory_access_t required_memory_access) {
  // TODO(scotttodd): change to allowed_memory_access when possible
  //   (at least the web samples aren't setting the required access bits)
  // if (iree_all_bits_set(allowed_memory_access, IREE_HAL_MEMORY_ACCESS_ANY)) {
  if (iree_all_bits_set(required_memory_access, IREE_HAL_MEMORY_ACCESS_ANY)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!iree_any_bit_set(
          required_memory_access,
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE))) {
    // No actual access bits defined.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "memory access must specify one or more of _READ or _WRITE");
  } else if (IREE_UNLIKELY(!iree_all_bits_set(allowed_memory_access,
                                              required_memory_access))) {
#if IREE_STATUS_MODE
    // Bits must match exactly.
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_memory_access_str =
        iree_hal_memory_access_format(allowed_memory_access, &temp0);
    iree_string_view_t required_memory_access_str =
        iree_hal_memory_access_format(required_memory_access, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "buffer does not support the requested access "
        "type; buffer allows %.*s, operation requires %.*s",
        (int)allowed_memory_access_str.size, allowed_memory_access_str.data,
        (int)required_memory_access_str.size, required_memory_access_str.data);
#else
    return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_validate_usage(iree_hal_buffer_usage_t allowed_usage,
                               iree_hal_buffer_usage_t required_usage) {
  if (IREE_UNLIKELY(!iree_all_bits_set(allowed_usage, required_usage))) {
#if IREE_STATUS_MODE
    // Missing one or more bits.
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_usage_str =
        iree_hal_buffer_usage_format(allowed_usage, &temp0);
    iree_string_view_t required_usage_str =
        iree_hal_buffer_usage_format(required_usage, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "requested usage was not specified when the buffer was allocated; "
        "buffer allows %.*s, operation requires %.*s",
        (int)allowed_usage_str.size, allowed_usage_str.data,
        (int)required_usage_str.size, required_usage_str.data);
#else
    return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length) {
  // Check if the start of the range runs off the end of the buffer.
  if (IREE_UNLIKELY(byte_offset > iree_hal_buffer_byte_length(buffer))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "attempted to access an address off the end of the valid buffer range "
        "(offset=%" PRIdsz ", length=%" PRIdsz ", buffer byte_length=%" PRIdsz
        ")",
        byte_offset, byte_length, iree_hal_buffer_byte_length(buffer));
  }

  if (byte_length == 0) {
    // Fine to have a zero length.
    return iree_ok_status();
  }

  // Check if the end runs over the allocation.
  iree_device_size_t end = byte_offset + byte_length;
  if (IREE_UNLIKELY(end > iree_hal_buffer_byte_length(buffer))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "attempted to access an address outside of the valid buffer range "
        "(offset=%" PRIdsz ", length=%" PRIdsz ", end(inc)=%" PRIdsz
        ", buffer byte_length=%" PRIdsz ")",
        byte_offset, byte_length, end - 1, iree_hal_buffer_byte_length(buffer));
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_buffer_calculate_range(
    iree_device_size_t base_offset, iree_device_size_t max_length,
    iree_device_size_t offset, iree_device_size_t length,
    iree_device_size_t* out_adjusted_offset,
    iree_device_size_t* out_adjusted_length) {
  // Check if the start of the range runs off the end of the buffer.
  if (IREE_UNLIKELY(offset > max_length)) {
    *out_adjusted_offset = 0;
    if (out_adjusted_length) *out_adjusted_length = 0;
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "attempted to access an address off the end of the valid buffer "
        "range (offset=%" PRIdsz ", length=%" PRIdsz
        ", buffer byte_length=%" PRIdsz ")",
        offset, length, max_length);
  }

  // Handle length as IREE_WHOLE_BUFFER by adjusting it (if allowed).
  if (IREE_UNLIKELY(length == IREE_WHOLE_BUFFER) &&
      IREE_UNLIKELY(!out_adjusted_length)) {
    *out_adjusted_offset = 0;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IREE_WHOLE_BUFFER may only be used with buffer "
                            "ranges, not external pointer ranges");
  }

  // Calculate the real ranges adjusted for our region within the allocation.
  iree_device_size_t adjusted_offset = base_offset + offset;
  iree_device_size_t adjusted_length =
      length == IREE_WHOLE_BUFFER ? max_length - offset : length;
  if (adjusted_length == 0) {
    // Fine to have a zero length.
    *out_adjusted_offset = adjusted_offset;
    if (out_adjusted_length) *out_adjusted_length = adjusted_length;
    return iree_ok_status();
  }

  // Check if the end runs over the allocation.
  iree_device_size_t end = offset + adjusted_length - 1;
  if (IREE_UNLIKELY(end >= max_length)) {
    *out_adjusted_offset = 0;
    if (out_adjusted_length) *out_adjusted_length = 0;
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "attempted to access an address outside of the valid buffer "
        "range (offset=%" PRIdsz ", adjusted_length=%" PRIdsz ", end=%" PRIdsz
        ", buffer byte_length=%" PRIdsz ")",
        offset, adjusted_length, end, max_length);
  }

  *out_adjusted_offset = adjusted_offset;
  if (out_adjusted_length) *out_adjusted_length = adjusted_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_buffer_overlap_t iree_hal_buffer_test_overlap(
    iree_hal_buffer_t* lhs_buffer, iree_device_size_t lhs_offset,
    iree_device_size_t lhs_length, iree_hal_buffer_t* rhs_buffer,
    iree_device_size_t rhs_offset, iree_device_size_t rhs_length) {
  if (iree_hal_buffer_allocated_buffer(lhs_buffer) !=
      iree_hal_buffer_allocated_buffer(rhs_buffer)) {
    // Not even the same buffers.
    return IREE_HAL_BUFFER_OVERLAP_DISJOINT;
  }
  // Resolve offsets into the underlying allocation.
  iree_device_size_t lhs_alloc_offset =
      iree_hal_buffer_byte_offset(lhs_buffer) + lhs_offset;
  iree_device_size_t rhs_alloc_offset =
      iree_hal_buffer_byte_offset(rhs_buffer) + rhs_offset;
  iree_device_size_t lhs_alloc_length =
      lhs_length == IREE_WHOLE_BUFFER
          ? iree_hal_buffer_byte_length(lhs_buffer) - lhs_offset
          : lhs_length;
  iree_device_size_t rhs_alloc_length =
      rhs_length == IREE_WHOLE_BUFFER
          ? iree_hal_buffer_byte_length(rhs_buffer) - rhs_offset
          : rhs_length;
  if (!lhs_alloc_length || !rhs_alloc_length) {
    return IREE_HAL_BUFFER_OVERLAP_DISJOINT;
  }
  if (lhs_alloc_offset == rhs_alloc_offset &&
      lhs_alloc_length == rhs_alloc_length) {
    return IREE_HAL_BUFFER_OVERLAP_COMPLETE;
  }
  return lhs_alloc_offset + lhs_alloc_length > rhs_alloc_offset &&
                 rhs_alloc_offset + rhs_alloc_length > lhs_alloc_offset
             ? IREE_HAL_BUFFER_OVERLAP_PARTIAL
             : IREE_HAL_BUFFER_OVERLAP_DISJOINT;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_subspan(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  // Fast path: if we are requesting the whole buffer (usually via
  // IREE_WHOLE_BUFFER) then we can just return the buffer itself.
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      iree_hal_buffer_byte_offset(buffer), iree_hal_buffer_byte_length(buffer),
      byte_offset, byte_length, &byte_offset, &byte_length));
  if (byte_offset == 0 && byte_length == iree_hal_buffer_byte_length(buffer)) {
    iree_hal_buffer_retain(buffer);
    *out_buffer = buffer;
    return iree_ok_status();
  }

  // To avoid heavy nesting of subspans that just add indirection we go to the
  // parent buffer directly. If we wanted better accounting (to track where
  // buffers came from) we'd want to avoid this but I'm not sure that's worth
  // the super deep indirection that could arise.
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (allocated_buffer != buffer) {
    return iree_hal_buffer_subspan(allocated_buffer, byte_offset, byte_length,
                                   out_buffer);
  }

  return iree_hal_subspan_buffer_create(buffer, byte_offset, byte_length,
                                        /*device_allocator=*/NULL,
                                        buffer->host_allocator, out_buffer);
}

IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_allocated_buffer(
    const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->allocated_buffer;
}

IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_allocation_size(const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->allocation_size;
}

IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_offset(const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->byte_offset;
}

IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_length(const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->byte_length;
}

IREE_API_EXPORT
iree_hal_memory_type_t iree_hal_buffer_memory_type(
    const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->memory_type;
}

IREE_API_EXPORT
iree_hal_memory_access_t iree_hal_buffer_allowed_access(
    const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->allowed_access;
}

IREE_API_EXPORT
iree_hal_buffer_usage_t iree_hal_buffer_allowed_usage(
    const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->allowed_usage;
}

//===----------------------------------------------------------------------===//
// Transfer
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_buffer_map_zero(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length) {
  const uint8_t zero = 0;
  return iree_hal_buffer_map_fill(buffer, byte_offset, byte_length, &zero, 1);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_map_fill(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, const void* pattern,
    iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(pattern);

  if (IREE_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                    pattern_length != 4)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fill patterns must be 1, 2, or 4 bytes (got %zu)",
                            pattern_length);
  }

  if (byte_length == 0) {
    return iree_ok_status();  // No-op.
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_buffer_mapping_t target_mapping = {{0}};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                    IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                    byte_offset, byte_length, &target_mapping));
  if (byte_length == IREE_WHOLE_BUFFER) {
    byte_length = target_mapping.contents.data_length;
  }

  if (IREE_UNLIKELY((byte_offset % pattern_length) != 0) ||
      IREE_UNLIKELY((byte_length % pattern_length) != 0)) {
    iree_status_ignore(iree_hal_buffer_unmap_range(&target_mapping));
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attempting to fill a range with %zu byte values "
                            "that is not aligned (offset=%" PRIdsz
                            ", length=%" PRIdsz ")",
                            pattern_length, byte_offset, byte_length);
  }

  const uint32_t zero_32 = 0;
  if (memcmp(pattern, &zero_32, pattern_length) == 0) {
    // We can turn all-zero values into single-byte fills as that can be much
    // faster on devices (doing a fill8 vs fill32).
    pattern_length = 1;
  }

  iree_status_t status = iree_ok_status();
  void* data_ptr = target_mapping.contents.data;
  switch (pattern_length) {
    case 1: {
      uint8_t* data = (uint8_t*)data_ptr;
      uint8_t value_bits = *(const uint8_t*)(pattern);
      memset(data, value_bits, byte_length);
      break;
    }
    case 2: {
      uint16_t* data = (uint16_t*)data_ptr;
      uint16_t value_bits = *(const uint16_t*)(pattern);
      for (iree_device_size_t i = 0; i < byte_length / sizeof(uint16_t); ++i) {
        data[i] = value_bits;
      }
      break;
    }
    case 4: {
      uint32_t* data = (uint32_t*)data_ptr;
      uint32_t value_bits = *(const uint32_t*)(pattern);
      for (iree_device_size_t i = 0; i < byte_length / sizeof(uint32_t); ++i) {
        data[i] = value_bits;
      }
      break;
    }
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported fill pattern length: %zu",
                                pattern_length);
      break;
  }

  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_flush_range(&target_mapping, 0,
                                                 IREE_WHOLE_BUFFER);
  }

  status =
      iree_status_join(status, iree_hal_buffer_unmap_range(&target_mapping));
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_map_read(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length) {
  if (data_length == 0) {
    return iree_ok_status();  // No-op.
  }
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, data_length);
  iree_hal_buffer_mapping_t source_mapping = {{0}};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_map_range(source_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                    IREE_HAL_MEMORY_ACCESS_READ, source_offset,
                                    data_length, &source_mapping));

  memcpy(target_buffer, source_mapping.contents.data, data_length);

  iree_hal_buffer_unmap_range(&source_mapping);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_map_write(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length) {
  if (data_length == 0) {
    return iree_ok_status();  // No-op.
  }
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_ASSERT_ARGUMENT(source_buffer);

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, data_length);
  iree_hal_buffer_mapping_t target_mapping;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_buffer_map_range(target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                target_offset, data_length, &target_mapping));

  memcpy(target_mapping.contents.data, source_buffer, data_length);

  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(iree_hal_buffer_memory_type(target_buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_flush_range(&target_mapping, 0,
                                                 IREE_WHOLE_BUFFER);
  }

  iree_hal_buffer_unmap_range(&target_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_map_copy(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t data_length) {
  if (data_length == 0) {
    return iree_ok_status();  // No-op.
  }
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);

  // Check for overlap - like memcpy we require that the two ranges don't have
  // any overlap - because we use memcpy below!
  if (iree_hal_buffer_test_overlap(source_buffer, source_offset, data_length,
                                   target_buffer, target_offset, data_length) !=
      IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, data_length);

  // Map source, which may have IREE_WHOLE_BUFFER length.
  iree_hal_buffer_mapping_t source_mapping;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_map_range(source_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                    IREE_HAL_MEMORY_ACCESS_READ, source_offset,
                                    data_length, &source_mapping));

  // Map target, which may also have IREE_WHOLE_BUFFER length.
  iree_hal_buffer_mapping_t target_mapping;
  iree_status_t status =
      iree_hal_buffer_map_range(target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                target_offset, data_length, &target_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_unmap_range(&source_mapping);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Adjust the data length based on the min we have.
  iree_device_size_t adjusted_data_length = 0;
  if (data_length == IREE_WHOLE_BUFFER) {
    // Whole buffer copy requested - that could mean either, so take the min.
    adjusted_data_length = iree_min(source_mapping.contents.data_length,
                                    target_mapping.contents.data_length);
  } else {
    // Specific length requested - validate that we have matching lengths.
    IREE_ASSERT_EQ(source_mapping.contents.data_length,
                   target_mapping.contents.data_length);
    adjusted_data_length = target_mapping.contents.data_length;
  }

  // Elide zero length copies. It's been expensive to get to this point just to
  // bail but we need to have mapped to resolve IREE_WHOLE_BUFFERs that may
  // result in zero lengths.
  if (IREE_UNLIKELY(adjusted_data_length == 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  memcpy(target_mapping.contents.data, source_mapping.contents.data,
         adjusted_data_length);

  if (!iree_all_bits_set(iree_hal_buffer_memory_type(target_buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_flush_range(&target_mapping, 0,
                                                 adjusted_data_length);
  }

  iree_hal_buffer_unmap_range(&source_mapping);
  iree_hal_buffer_unmap_range(&target_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Mapping
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_buffer_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length,
    iree_hal_buffer_mapping_t* out_buffer_mapping) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer_mapping);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_buffer_mapping, 0, sizeof(*out_buffer_mapping));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_validate_access(
              iree_hal_buffer_allowed_access(buffer), memory_access));

  // Persistent mapping requires the buffer was allocated to support it.
  const bool is_persistent =
      iree_all_bits_set(mapping_mode, IREE_HAL_MAPPING_MODE_PERSISTENT);
  if (is_persistent) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_buffer_validate_memory_type(
                                          iree_hal_buffer_memory_type(buffer),
                                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_buffer_validate_usage(iree_hal_buffer_allowed_usage(buffer),
                                       IREE_HAL_BUFFER_USAGE_MAPPING));
  }

  iree_device_size_t local_byte_offset = 0;
  iree_device_size_t local_byte_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_calculate_range(
              iree_hal_buffer_byte_offset(buffer),
              iree_hal_buffer_byte_length(buffer), byte_offset, byte_length,
              &local_byte_offset, &local_byte_length));

  out_buffer_mapping->buffer = buffer;
  out_buffer_mapping->impl.allowed_access = memory_access;
  out_buffer_mapping->impl.is_persistent = is_persistent ? 1 : 0;
  out_buffer_mapping->impl.byte_offset = local_byte_offset;

  iree_status_t status = _VTABLE_DISPATCH(buffer, map_range)(
      buffer, mapping_mode, memory_access, out_buffer_mapping->impl.byte_offset,
      local_byte_length, out_buffer_mapping);

  if (iree_status_is_ok(status)) {
    // Scoped mappings retain the buffer until unmapped.
    if (!is_persistent) iree_hal_buffer_retain(buffer);
  } else {
    memset(out_buffer_mapping, 0, sizeof(*out_buffer_mapping));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_unmap_range(iree_hal_buffer_mapping_t* buffer_mapping) {
  IREE_ASSERT_ARGUMENT(buffer_mapping);
  iree_hal_buffer_t* buffer = buffer_mapping->buffer;
  if (!buffer) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = _VTABLE_DISPATCH(buffer, unmap_range)(
      buffer, buffer_mapping->impl.byte_offset,
      buffer_mapping->contents.data_length, buffer_mapping);

  if (!buffer_mapping->impl.is_persistent) {
    iree_hal_buffer_release(buffer);
  }
  memset(buffer_mapping, 0, sizeof(*buffer_mapping));

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_invalidate_range(
    iree_hal_buffer_mapping_t* buffer_mapping, iree_device_size_t byte_offset,
    iree_device_size_t byte_length) {
  IREE_ASSERT_ARGUMENT(buffer_mapping);
  iree_hal_buffer_t* buffer = buffer_mapping->buffer;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      buffer_mapping->impl.allowed_access, IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      buffer_mapping->impl.byte_offset, buffer_mapping->contents.data_length,
      byte_offset, byte_length, &byte_offset, &byte_length));
  return _VTABLE_DISPATCH(buffer, invalidate_range)(buffer, byte_offset,
                                                    byte_length);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_flush_range(
    iree_hal_buffer_mapping_t* buffer_mapping, iree_device_size_t byte_offset,
    iree_device_size_t byte_length) {
  IREE_ASSERT_ARGUMENT(buffer_mapping);
  iree_hal_buffer_t* buffer = buffer_mapping->buffer;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      buffer_mapping->impl.allowed_access, IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      buffer_mapping->impl.byte_offset, buffer_mapping->contents.data_length,
      byte_offset, byte_length, &byte_offset, &byte_length));
  return _VTABLE_DISPATCH(buffer, flush_range)(buffer, byte_offset,
                                               byte_length);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_subspan(
    iree_hal_buffer_mapping_t* buffer_mapping,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_byte_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(buffer_mapping);
  IREE_ASSERT_ARGUMENT(out_span);
  memset(out_span, 0, sizeof(*out_span));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      buffer_mapping->impl.allowed_access, memory_access));
  iree_device_size_t data_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      0, buffer_mapping->contents.data_length, byte_offset, byte_length,
      &byte_offset, &data_length));
  out_span->data_length = data_length;
  out_span->data = buffer_mapping->contents.data + byte_offset;
  return iree_ok_status();
}
