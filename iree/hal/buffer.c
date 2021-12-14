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
      {IREE_HAL_MEMORY_TYPE_TRANSIENT, IREE_SVL("TRANSIENT")},
      {IREE_HAL_MEMORY_TYPE_HOST_VISIBLE, IREE_SVL("HOST_VISIBLE")},
      {IREE_HAL_MEMORY_TYPE_HOST_COHERENT, IREE_SVL("HOST_COHERENT")},
      {IREE_HAL_MEMORY_TYPE_HOST_CACHED, IREE_SVL("HOST_CACHED")},
      {IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_SVL("DEVICE_VISIBLE")},
  };
  return iree_bitfield_format_inline(value, mappings, IREE_ARRAYSIZE(mappings),
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
  return iree_bitfield_format_inline(value, mappings, IREE_ARRAYSIZE(mappings),
                                     out_temp);
}

IREE_API_EXPORT iree_string_view_t iree_hal_buffer_usage_format(
    iree_hal_buffer_usage_t value, iree_bitfield_string_temp_t* out_temp) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      // Combined:
      {IREE_HAL_BUFFER_USAGE_ALL, IREE_SVL("ALL")},
      // Separate:
      {IREE_HAL_BUFFER_USAGE_CONSTANT, IREE_SVL("CONSTANT")},
      {IREE_HAL_BUFFER_USAGE_TRANSFER, IREE_SVL("TRANSFER")},
      {IREE_HAL_BUFFER_USAGE_MAPPING, IREE_SVL("MAPPING")},
      {IREE_HAL_BUFFER_USAGE_DISPATCH, IREE_SVL("DISPATCH")},
  };
  return iree_bitfield_format_inline(value, mappings, IREE_ARRAYSIZE(mappings),
                                     out_temp);
}

//===----------------------------------------------------------------------===//
// Subspan indirection buffer
//===----------------------------------------------------------------------===//

static const iree_hal_buffer_vtable_t iree_hal_subspan_buffer_vtable;

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
    void** out_data_ptr) {
  return _VTABLE_DISPATCH(buffer->allocated_buffer, map_range)(
      buffer->allocated_buffer, mapping_mode, memory_access, local_byte_offset,
      local_byte_length, out_data_ptr);
}

static void iree_hal_subspan_buffer_unmap_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, void* data_ptr) {
  if (!buffer->allocated_buffer) return;
  _VTABLE_DISPATCH(buffer->allocated_buffer, unmap_range)
  (buffer->allocated_buffer, local_byte_offset, local_byte_length, data_ptr);
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
    // If the buffer comes from an allocator then we route back the destruction
    // request to that. It may decide to keep the buffer alive in a pool or
    // do some allocator-specific cleanup.
    if (buffer->device_allocator) {
      iree_hal_allocator_deallocate_buffer(buffer->device_allocator, buffer);
    } else {
      iree_hal_buffer_destroy(buffer);
    }
  }
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_memory_type(
    iree_hal_memory_type_t actual_memory_type,
    iree_hal_memory_type_t expected_memory_type) {
  if (IREE_UNLIKELY(
          !iree_all_bits_set(actual_memory_type, expected_memory_type))) {
    // Missing one or more bits.
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t actual_memory_type_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_memory_type_format(actual_memory_type, &temp0);
    iree_string_view_t expected_memory_type_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_memory_type_format(expected_memory_type, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "buffer memory type is not compatible with the requested operation; "
        "buffer has %.*s, operation requires %.*s",
        (int)actual_memory_type_str.size, actual_memory_type_str.data,
        (int)expected_memory_type_str.size, expected_memory_type_str.data);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_access(
    iree_hal_memory_access_t allowed_memory_access,
    iree_hal_memory_access_t required_memory_access) {
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
    // Bits must match exactly.
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_memory_access_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_memory_access_format(allowed_memory_access, &temp0);
    iree_string_view_t required_memory_access_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_memory_access_format(required_memory_access, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "buffer does not support the requested access "
        "type; buffer allows %.*s, operation requires %.*s",
        (int)allowed_memory_access_str.size, allowed_memory_access_str.data,
        (int)required_memory_access_str.size, required_memory_access_str.data);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_validate_usage(iree_hal_buffer_usage_t allowed_usage,
                               iree_hal_buffer_usage_t required_usage) {
  if (IREE_UNLIKELY(!iree_all_bits_set(allowed_usage, required_usage))) {
    // Missing one or more bits.
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_usage_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_buffer_usage_format(allowed_usage, &temp0);
    iree_string_view_t required_usage_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_buffer_usage_format(required_usage, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "requested usage was not specified when the buffer was allocated; "
        "buffer allows %.*s, operation requires %.*s",
        (int)allowed_usage_str.size, allowed_usage_str.data,
        (int)required_usage_str.size, required_usage_str.data);
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

IREE_API_EXPORT iree_status_t
iree_hal_buffer_zero(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length) {
  const uint8_t zero = 0;
  return iree_hal_buffer_fill(buffer, byte_offset, byte_length, &zero, 1);
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_fill(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
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
  iree_hal_buffer_mapping_t target_mapping;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                    IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                    byte_offset, byte_length, &target_mapping));
  if (byte_length == IREE_WHOLE_BUFFER) {
    byte_length = target_mapping.contents.data_length;
  }

  if (IREE_UNLIKELY((byte_offset % pattern_length) != 0) ||
      IREE_UNLIKELY((byte_length % pattern_length) != 0)) {
    iree_hal_buffer_unmap_range(&target_mapping);
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
    status = iree_hal_buffer_flush_range(&target_mapping, 0, IREE_WHOLE_BUFFER);
  }

  iree_hal_buffer_unmap_range(&target_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_read_data(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length) {
  return iree_hal_buffer_transfer_range(
      iree_hal_make_device_transfer_buffer(source_buffer), source_offset,
      iree_hal_make_host_transfer_buffer_span(target_buffer, data_length), 0,
      data_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_write_data(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length) {
  return iree_hal_buffer_transfer_range(
      iree_hal_make_host_transfer_buffer_span((void*)source_buffer,
                                              data_length),
      0, iree_hal_make_device_transfer_buffer(target_buffer), target_offset,
      data_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_copy_data(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t data_length) {
  return iree_hal_buffer_transfer_range(
      iree_hal_make_device_transfer_buffer(source_buffer), source_offset,
      iree_hal_make_device_transfer_buffer(target_buffer), target_offset,
      data_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_transfer_range(
    iree_hal_transfer_buffer_t source, iree_device_size_t source_offset,
    iree_hal_transfer_buffer_t target, iree_device_size_t target_offset,
    iree_device_size_t data_length, iree_hal_transfer_buffer_flags_t flags) {
  if (data_length == 0) {
    return iree_ok_status();  // No-op.
  }

  // host->host is not allowed. We may want to support this one day to allow for
  // parallelized copies and such, however the validation code differs quite a
  // bit and it'd be better to have this as part of a task system API.
  bool is_source_host = source.device_buffer == NULL;
  bool is_target_host = target.device_buffer == NULL;
  if (is_source_host && is_target_host) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot perform host->host transfers via this API, use memcpy/memmove");
  }

  // Check for overlap - like memcpy we require that the two ranges don't have
  // any overlap as we may use memcpy. This only matters if the buffers are
  // both device buffers - host and device should never alias: behavior is
  // undefined if a user tries to pass a mapped device pointer as if it was a
  // host pointer.
  if (!is_source_host && !is_target_host &&
      iree_hal_buffer_test_overlap(source.device_buffer, source_offset,
                                   data_length, target.device_buffer,
                                   target_offset, data_length) !=
          IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, is_source_host ? "h2d" : (is_target_host ? "d2h" : "d2d"));
  IREE_TRACE_ZONE_APPEND_VALUE(z0, data_length);

  // Defer to the backing implementation.
  // We prefer the target as that's likely to be where the data is used next.
  iree_hal_buffer_t* vtable_buffer =
      !is_target_host ? target.device_buffer : source.device_buffer;
  iree_status_t status = _VTABLE_DISPATCH(vtable_buffer, transfer_range)(
      source, source_offset, target, target_offset, data_length, flags);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_transfer_mappable_range(
    iree_hal_transfer_buffer_t source, iree_device_size_t source_offset,
    iree_hal_transfer_buffer_t target, iree_device_size_t target_offset,
    iree_device_size_t data_length, iree_hal_transfer_buffer_flags_t flags) {
  iree_status_t status = iree_ok_status();

  iree_hal_buffer_mapping_t source_mapping = {0};
  if (iree_status_is_ok(status)) {
    if (source.device_buffer) {
      status = iree_hal_buffer_map_range(
          source.device_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
          IREE_HAL_MEMORY_ACCESS_READ, source_offset, data_length,
          &source_mapping);
    } else {
      source_mapping = (iree_hal_buffer_mapping_t){
          .contents = source.host_buffer,
      };
    }
  }

  iree_hal_buffer_mapping_t target_mapping = {0};
  if (iree_status_is_ok(status)) {
    if (target.device_buffer) {
      status = iree_hal_buffer_map_range(
          target.device_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
          IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, target_offset, data_length,
          &target_mapping);
    } else {
      target_mapping = (iree_hal_buffer_mapping_t){
          .contents = target.host_buffer,
      };
    }
  }

  iree_device_size_t adjusted_data_length = 0;
  if (iree_status_is_ok(status)) {
    // Adjust the data length based on the min we have.
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

    // Perform the copy, assuming there's anything to do.
    if (adjusted_data_length != 0) {
      memcpy(target_mapping.contents.data, source_mapping.contents.data,
             adjusted_data_length);
    }
  }

  if (source.device_buffer) {
    iree_hal_buffer_unmap_range(&source_mapping);
  }
  if (target.device_buffer) {
    if (adjusted_data_length > 0 &&
        !iree_all_bits_set(iree_hal_buffer_memory_type(target.device_buffer),
                           IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
      status = iree_status_join(
          status, iree_hal_buffer_flush_range(&target_mapping, 0,
                                              adjusted_data_length));
    }
    iree_hal_buffer_unmap_range(&target_mapping);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Mapping / iree_hal_buffer_mapping_impl_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_buffer_mapping_impl_t {
  // Must be first (as in iree_hal_buffer_mapping_t).
  // Stores both the offset data pointer and the byte_length of the mapping.
  iree_byte_span_t contents;
  // Retained buffer providing the backing storage for the mapping.
  iree_hal_buffer_t* backing_buffer;
  // Byte offset within the buffer where the mapped data begins.
  iree_device_size_t byte_offset;
  // Used for validation only.
  iree_hal_memory_access_t allowed_access;
  uint32_t reserved0;  // unused
  uint64_t reserved1;  // unused
} iree_hal_buffer_mapping_impl_t;

// We overlay the impl onto the external iree_hal_buffer_mapping_t struct;
// ensure we match the fields that are exposed.
static_assert(sizeof(iree_hal_buffer_mapping_impl_t) <=
                  sizeof(iree_hal_buffer_mapping_t),
              "buffer mapping impl must fit inside the external struct");
static_assert(offsetof(iree_hal_buffer_mapping_impl_t, contents) ==
                  offsetof(iree_hal_buffer_mapping_t, contents),
              "contents byte span must match the external struct offset");

IREE_API_EXPORT iree_status_t iree_hal_buffer_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length,
    iree_hal_buffer_mapping_t* out_buffer_mapping) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer_mapping);
  memset(out_buffer_mapping, 0, sizeof(*out_buffer_mapping));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer), memory_access));

  // Persistent mapping requires the buffer be allocated to support it.
  if (iree_all_bits_set(mapping_mode, IREE_HAL_MAPPING_MODE_PERSISTENT)) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(buffer),
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
        iree_hal_buffer_allowed_usage(buffer), IREE_HAL_BUFFER_USAGE_MAPPING));
  }

  iree_hal_buffer_mapping_impl_t* buffer_mapping =
      (iree_hal_buffer_mapping_impl_t*)out_buffer_mapping;
  buffer_mapping->backing_buffer = buffer;
  buffer_mapping->allowed_access = memory_access;
  iree_device_size_t data_length;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      iree_hal_buffer_byte_offset(buffer), iree_hal_buffer_byte_length(buffer),
      byte_offset, byte_length, &buffer_mapping->byte_offset, &data_length));
  buffer_mapping->contents.data_length = data_length;

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(buffer, map_range)(
      buffer, mapping_mode, buffer_mapping->allowed_access,
      buffer_mapping->byte_offset, buffer_mapping->contents.data_length,
      (void**)&buffer_mapping->contents.data);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_buffer_unmap_range(
    iree_hal_buffer_mapping_t* base_buffer_mapping) {
  IREE_ASSERT_ARGUMENT(base_buffer_mapping);
  iree_hal_buffer_mapping_impl_t* buffer_mapping =
      (iree_hal_buffer_mapping_impl_t*)base_buffer_mapping;
  iree_hal_buffer_t* buffer = buffer_mapping->backing_buffer;
  if (!buffer) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  _VTABLE_DISPATCH(buffer, unmap_range)
  (buffer, buffer_mapping->byte_offset, buffer_mapping->contents.data_length,
   buffer_mapping->contents.data);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_invalidate_range(
    iree_hal_buffer_mapping_t* base_buffer_mapping,
    iree_device_size_t byte_offset, iree_device_size_t byte_length) {
  IREE_ASSERT_ARGUMENT(base_buffer_mapping);
  iree_hal_buffer_mapping_impl_t* buffer_mapping =
      (iree_hal_buffer_mapping_impl_t*)base_buffer_mapping;
  iree_hal_buffer_t* buffer = buffer_mapping->backing_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      buffer_mapping->allowed_access, IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      buffer_mapping->byte_offset, buffer_mapping->contents.data_length,
      byte_offset, byte_length, &byte_offset, &byte_length));
  return _VTABLE_DISPATCH(buffer, invalidate_range)(buffer, byte_offset,
                                                    byte_length);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_flush_range(
    iree_hal_buffer_mapping_t* base_buffer_mapping,
    iree_device_size_t byte_offset, iree_device_size_t byte_length) {
  IREE_ASSERT_ARGUMENT(base_buffer_mapping);
  iree_hal_buffer_mapping_impl_t* buffer_mapping =
      (iree_hal_buffer_mapping_impl_t*)base_buffer_mapping;
  iree_hal_buffer_t* buffer = buffer_mapping->backing_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      buffer_mapping->allowed_access, IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      buffer_mapping->byte_offset, buffer_mapping->contents.data_length,
      byte_offset, byte_length, &byte_offset, &byte_length));
  return _VTABLE_DISPATCH(buffer, flush_range)(buffer, byte_offset,
                                               byte_length);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_subspan(
    iree_hal_buffer_mapping_t* base_buffer_mapping,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_byte_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(base_buffer_mapping);
  iree_hal_buffer_mapping_impl_t* buffer_mapping =
      (iree_hal_buffer_mapping_impl_t*)base_buffer_mapping;
  IREE_ASSERT_ARGUMENT(out_span);
  memset(out_span, 0, sizeof(*out_span));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      buffer_mapping->allowed_access, memory_access));
  iree_device_size_t data_length;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      0, buffer_mapping->contents.data_length, byte_offset, byte_length,
      &byte_offset, &data_length));
  out_span->data_length = data_length;
  out_span->data = buffer_mapping->contents.data + byte_offset;
  return iree_ok_status();
}
