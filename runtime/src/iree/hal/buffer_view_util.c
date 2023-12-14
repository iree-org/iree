// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/buffer_view_util.h"

#include <inttypes.h>
#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/hal/buffer_transfer.h"
#include "iree/hal/resource.h"
#include "iree/hal/string_util.h"

//===----------------------------------------------------------------------===//
// Buffer view math
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_size(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_device_size_t* out_allocation_size) {
  IREE_ASSERT_ARGUMENT(!shape_rank || shape);
  IREE_ASSERT_ARGUMENT(out_allocation_size);
  *out_allocation_size = 0;

  iree_device_size_t byte_length = 0;

  switch (encoding_type) {
    case IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR: {
      if (IREE_UNLIKELY(iree_hal_element_bit_count(element_type) == 0) ||
          IREE_UNLIKELY(!iree_hal_element_is_byte_aligned(element_type))) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "opaque and sub-byte aligned element types cannot be indexed");
      }
      byte_length = iree_hal_element_dense_byte_count(element_type);
      for (iree_host_size_t i = 0; i < shape_rank; ++i) {
        byte_length *= shape[i];
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented encoding type size calculation");
  }

  *out_allocation_size = byte_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_offset(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_host_size_t indices_count,
    const iree_hal_dim_t* indices, iree_device_size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(!shape_rank || shape);
  IREE_ASSERT_ARGUMENT(!indices_count || indices);
  IREE_ASSERT_ARGUMENT(out_offset);
  *out_offset = 0;
  if (IREE_UNLIKELY(encoding_type != IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "only dense encodings support view range computation");
  } else if (IREE_UNLIKELY(iree_hal_element_bit_count(element_type) == 0) ||
             IREE_UNLIKELY(!iree_hal_element_is_byte_aligned(element_type))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "opaque and sub-byte aligned element types cannot be indexed");
  } else if (IREE_UNLIKELY(shape_rank != indices_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shape rank/indices mismatch: %" PRIhsz
                            " != %" PRIhsz,
                            shape_rank, indices_count);
  }

  iree_device_size_t offset = 0;
  for (iree_host_size_t i = 0; i < indices_count; ++i) {
    if (IREE_UNLIKELY(indices[i] >= shape[i])) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "index[%" PRIhsz "] out of bounds: %" PRIdim
                              " >= %" PRIdim,
                              i, indices[i], shape[i]);
    }
    iree_device_size_t axis_offset = indices[i];
    for (iree_host_size_t j = i + 1; j < shape_rank; ++j) {
      axis_offset *= shape[j];
    }
    offset += axis_offset;
  }
  offset *= iree_hal_element_dense_byte_count(element_type);

  *out_offset = offset;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_range(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_host_size_t indices_count,
    const iree_hal_dim_t* start_indices, iree_host_size_t lengths_count,
    const iree_hal_dim_t* lengths, iree_device_size_t* out_start_offset,
    iree_device_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(!shape_rank || shape);
  IREE_ASSERT_ARGUMENT(!indices_count || start_indices);
  IREE_ASSERT_ARGUMENT(!lengths_count || lengths);
  IREE_ASSERT_ARGUMENT(out_start_offset);
  IREE_ASSERT_ARGUMENT(out_length);
  *out_start_offset = 0;
  *out_length = 0;
  if (IREE_UNLIKELY(encoding_type != IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "only dense encodings support view range computation");
  } else if (IREE_UNLIKELY(iree_hal_element_bit_count(element_type) == 0) ||
             IREE_UNLIKELY(!iree_hal_element_is_byte_aligned(element_type))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "opaque and sub-byte aligned element types cannot be indexed");
  } else if (IREE_UNLIKELY(indices_count != lengths_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indices/lengths mismatch: %" PRIhsz " != %" PRIhsz,
                            indices_count, lengths_count);
  } else if (IREE_UNLIKELY(shape_rank != indices_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shape rank/indices mismatch: %" PRIhsz
                            " != %" PRIhsz,
                            shape_rank, indices_count);
  }

  iree_hal_dim_t* end_indices =
      iree_alloca(shape_rank * sizeof(iree_hal_dim_t));
  iree_device_size_t element_size =
      iree_hal_element_dense_byte_count(element_type);
  iree_device_size_t subspan_length = element_size;
  for (iree_host_size_t i = 0; i < lengths_count; ++i) {
    subspan_length *= lengths[i];
    end_indices[i] = start_indices[i] + lengths[i] - 1;
  }

  iree_device_size_t start_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_compute_view_offset(
      shape_rank, shape, element_type, encoding_type, shape_rank, start_indices,
      &start_byte_offset));
  iree_device_size_t end_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_compute_view_offset(
      shape_rank, shape, element_type, encoding_type, shape_rank, end_indices,
      &end_byte_offset));

  // Non-contiguous regions not yet implemented. Will be easier to detect when
  // we have strides.
  iree_device_size_t offset_length =
      end_byte_offset - start_byte_offset + element_size;
  if (subspan_length != offset_length) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-contiguous range region computation not implemented");
  }

  *out_start_offset = start_byte_offset;
  *out_length = subspan_length;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Buffer view allocation and generation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_allocate_buffer_copy(
    iree_hal_device_t* device, iree_hal_allocator_t* allocator,
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_params_t buffer_params, iree_const_byte_span_t initial_data,
    iree_hal_buffer_view_t** out_buffer_view) {
  if (!iree_const_byte_span_is_empty(initial_data) && !device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "a device is required for initial_data upload");
  }
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer_view);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_buffer_params_canonicalize(&buffer_params);

  iree_device_size_t allocation_size = 0;
  iree_status_t status = iree_hal_buffer_compute_view_size(
      shape_rank, shape, element_type, encoding_type, &allocation_size);

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(allocator, buffer_params,
                                                allocation_size, &buffer);
  }
  if (iree_status_is_ok(status) &&
      !iree_const_byte_span_is_empty(initial_data)) {
    status = iree_hal_device_transfer_h2d(
        device, initial_data.data, buffer, 0,
        iree_min(initial_data.data_length, allocation_size),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_create(
        buffer, shape_rank, shape, element_type, encoding_type,
        iree_hal_allocator_host_allocator(allocator), out_buffer_view);
  }

  iree_hal_buffer_release(buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_buffer_view_generate_buffer_in_situ(
    iree_hal_device_t* device, iree_hal_allocator_t* allocator,
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_params_t buffer_params,
    iree_hal_buffer_view_generator_callback_t callback, void* user_data,
    iree_hal_buffer_view_t** out_buffer_view) {
  // Allocate the buffer view and entire buffer contents with the target memory
  // type and the mapping bits.
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, shape_rank, shape, element_type, encoding_type,
      iree_hal_buffer_params_with_usage(buffer_params,
                                        IREE_HAL_BUFFER_USAGE_MAPPING),
      iree_const_byte_span_empty(), &buffer_view));

  // Map the buffer into host-visible memory.
  iree_hal_buffer_mapping_t buffer_mapping = {{0}};
  iree_status_t status = iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(buffer_view), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0, IREE_WHOLE_BUFFER,
      &buffer_mapping);

  // Generate using the callback directly into the buffer.
  if (iree_status_is_ok(status)) {
    status = callback(&buffer_mapping, user_data);
  }

  status =
      iree_status_join(status, iree_hal_buffer_unmap_range(&buffer_mapping));
  if (iree_status_is_ok(status)) {
    *out_buffer_view = buffer_view;
  } else {
    iree_hal_buffer_view_release(buffer_view);
  }
  return status;
}

static iree_status_t iree_hal_buffer_view_generate_buffer_on_host(
    iree_hal_device_t* device, iree_hal_allocator_t* allocator,
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_params_t buffer_params, iree_device_size_t allocation_size,
    iree_hal_buffer_view_generator_callback_t callback, void* user_data,
    iree_hal_buffer_view_t** out_buffer_view) {
  // Allocate the host memory and generate the contents.
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(allocator);
  void* host_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, &host_ptr));
  iree_hal_buffer_mapping_t mapping = {
      .contents = iree_make_byte_span(host_ptr, allocation_size),
  };
  iree_status_t status = callback(&mapping, user_data);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, host_ptr);
    return status;
  }

  // Allocate the buffer with the data we just generated.
  // We could try importing but that may create buffers that are slower to
  // access and we want users to opt in to that instead.
  status = iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, shape_rank, shape, element_type, encoding_type,
      buffer_params, iree_make_const_byte_span(host_ptr, allocation_size),
      out_buffer_view);

  iree_allocator_free(host_allocator, host_ptr);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_generate_buffer(
    iree_hal_device_t* device, iree_hal_allocator_t* allocator,
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_params_t buffer_params,
    iree_hal_buffer_view_generator_callback_t callback, void* user_data,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(callback);
  IREE_ASSERT_ARGUMENT(out_buffer_view);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_buffer_params_canonicalize(&buffer_params);

  // Compute how large of an allocation we need to hold the whole view.
  iree_device_size_t allocation_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_compute_view_size(shape_rank, shape, element_type,
                                            encoding_type, &allocation_size));

  // If we can create the requested memory type with mapping then we'll do that
  // and avoid needing to allocate the staging memory. If we can't get that
  // memory type (or the allocator doesn't want us using it) then we'll fall
  // back to allocation -> generation -> copy.
  iree_hal_buffer_params_t mappable_params = buffer_params;
  mappable_params.type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  mappable_params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          allocator, mappable_params, allocation_size, &mappable_params,
          &allocation_size);
  bool is_mappable =
      iree_all_bits_set(compatibility,
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE) &&
      !iree_any_bit_set(compatibility,
                        IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE);

  iree_status_t status = iree_ok_status();
  if (is_mappable) {
    // Compatible with allocate -> map -> generate.
    status = iree_hal_buffer_view_generate_buffer_in_situ(
        device, allocator, shape_rank, shape, element_type, encoding_type,
        mappable_params, callback, user_data, out_buffer_view);
  } else {
    // Allocate host-local memory first and generate into that.
    status = iree_hal_buffer_view_generate_buffer_on_host(
        device, allocator, shape_rank, shape, element_type, encoding_type,
        buffer_params, allocation_size, callback, user_data, out_buffer_view);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Buffer view parsing and printing
//===----------------------------------------------------------------------===//

typedef struct iree_hal_buffer_view_parse_params_t {
  iree_string_view_t data_str;
  iree_hal_element_type_t element_type;
} iree_hal_buffer_view_parse_params_t;
static iree_status_t iree_hal_buffer_view_parse_into(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  iree_hal_buffer_view_parse_params_t* params =
      (iree_hal_buffer_view_parse_params_t*)user_data;
  return iree_hal_parse_buffer_elements(params->data_str, params->element_type,
                                        mapping->contents);
}

static iree_status_t iree_hal_buffer_view_parse_impl(
    iree_string_view_t value, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  // Strip whitespace that may come along (linefeeds/etc).
  value = iree_string_view_trim(value);
  value = iree_string_view_strip_prefix(value, IREE_SV("\""));
  value = iree_string_view_strip_suffix(value, IREE_SV("\""));
  if (iree_string_view_is_empty(value)) {
    // Empty lines are invalid; need at least the shape/type information.
    *out_buffer_view = NULL;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty string input");
  }

  // The part of the string corresponding to the shape, e.g. 1x2x3.
  iree_string_view_t shape_str = iree_string_view_empty();
  // The part of the string corresponding to the type, e.g. f32
  iree_string_view_t type_str = iree_string_view_empty();
  // The part of the string corresponding to the buffer data, e.g. 1 2 3 4 5 6
  iree_string_view_t data_str = iree_string_view_empty();

  iree_string_view_t shape_and_type_str = value;
  iree_string_view_split(value, '=', &shape_and_type_str, &data_str);
  iree_host_size_t last_x_index = iree_string_view_find_last_of(
      shape_and_type_str, IREE_SV("x"), IREE_STRING_VIEW_NPOS);
  if (last_x_index == IREE_STRING_VIEW_NPOS) {
    // Scalar.
    type_str = shape_and_type_str;
  } else {
    // Has a shape.
    shape_str = iree_string_view_substr(shape_and_type_str, 0, last_x_index);
    type_str = iree_string_view_substr(shape_and_type_str, last_x_index + 1,
                                       IREE_STRING_VIEW_NPOS);
  }

  // AxBxC...
  iree_host_size_t shape_rank = 0;
  iree_status_t shape_result =
      iree_hal_parse_shape(shape_str, 0, &shape_rank, NULL);
  if (!iree_status_is_ok(shape_result) &&
      !iree_status_is_out_of_range(shape_result)) {
    return shape_result;
  } else if (shape_rank > 128) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "a shape rank of %" PRIhsz
                            " is just a little bit excessive, eh?",
                            shape_rank);
  }
  shape_result = iree_status_ignore(shape_result);
  iree_hal_dim_t* shape =
      (iree_hal_dim_t*)iree_alloca(shape_rank * sizeof(iree_hal_dim_t));
  IREE_RETURN_IF_ERROR(
      iree_hal_parse_shape(shape_str, shape_rank, &shape_rank, shape));

  // f32, i32, etc
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_parse_element_type(type_str, &element_type));

  // TODO(benvanik): allow specifying the encoding.
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;

  // Allocate the buffer from the provided allocator and parse directly into it.
  const iree_hal_buffer_params_t buffer_params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  iree_hal_buffer_view_parse_params_t parse_params = {
      .data_str = data_str,
      .element_type = element_type,
  };
  return iree_hal_buffer_view_generate_buffer(
      device, device_allocator, shape_rank, shape, element_type, encoding_type,
      buffer_params, iree_hal_buffer_view_parse_into, &parse_params,
      out_buffer_view);
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_view_parse(iree_string_view_t value, iree_hal_device_t* device,
                           iree_hal_allocator_t* device_allocator,
                           iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_buffer_view);
  *out_buffer_view = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_buffer_view_parse_impl(
      value, device, device_allocator, out_buffer_view);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#define APPEND_CHAR(c)                           \
  {                                              \
    if (buffer) {                                \
      if (buffer_length < buffer_capacity - 1) { \
        buffer[buffer_length] = c;               \
        buffer[buffer_length + 1] = '\0';        \
      } else {                                   \
        buffer = NULL;                           \
      }                                          \
    }                                            \
    ++buffer_length;                             \
  }

static iree_status_t iree_hal_buffer_view_format_impl(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  if (out_buffer_length) {
    *out_buffer_length = 0;
  }
  if (buffer && buffer_capacity) {
    buffer[0] = 0;
  }

  iree_host_size_t buffer_length = 0;
  if (iree_hal_buffer_view_shape_rank(buffer_view) > 0) {
    // Shape: 1x2x3
    iree_host_size_t shape_length = 0;
    iree_status_t status = iree_hal_format_shape(
        iree_hal_buffer_view_shape_rank(buffer_view),
        iree_hal_buffer_view_shape_dims(buffer_view),
        buffer ? buffer_capacity - buffer_length : 0,
        buffer ? buffer + buffer_length : NULL, &shape_length);
    buffer_length += shape_length;
    if (iree_status_is_out_of_range(status)) {
      status = iree_status_ignore(status);
      buffer = NULL;
    } else if (!iree_status_is_ok(status)) {
      return status;
    }

    // Separator: <shape>x<format>
    APPEND_CHAR('x');
  }

  // Element type: f32
  iree_host_size_t element_type_length = 0;
  iree_status_t status = iree_hal_format_element_type(
      iree_hal_buffer_view_element_type(buffer_view),
      buffer ? buffer_capacity - buffer_length : 0,
      buffer ? buffer + buffer_length : NULL, &element_type_length);
  buffer_length += element_type_length;
  if (iree_status_is_out_of_range(status)) {
    status = iree_status_ignore(status);
    buffer = NULL;
  } else if (!iree_status_is_ok(status)) {
    return status;
  }

  // TODO(benvanik): allow printing the encoding.

  // Separator: <meta>=<value>
  APPEND_CHAR('=');

  // Buffer contents: 0 1 2 3 ...
  iree_hal_buffer_mapping_t buffer_mapping = {{0}};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(buffer_view), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_WHOLE_BUFFER, &buffer_mapping));
  iree_host_size_t elements_length = 0;
  status = iree_hal_format_buffer_elements(
      iree_make_const_byte_span(buffer_mapping.contents.data,
                                buffer_mapping.contents.data_length),
      iree_hal_buffer_view_shape_rank(buffer_view),
      iree_hal_buffer_view_shape_dims(buffer_view),
      iree_hal_buffer_view_element_type(buffer_view), max_element_count,
      buffer ? buffer_capacity - buffer_length : 0,
      buffer ? buffer + buffer_length : NULL, &elements_length);
  buffer_length += elements_length;
  status =
      iree_status_join(status, iree_hal_buffer_unmap_range(&buffer_mapping));
  if (iree_status_is_out_of_range(status)) {
    status = iree_status_ignore(status);
    buffer = NULL;
  } else if (!iree_status_is_ok(status)) {
    return status;
  }

  if (out_buffer_length) {
    *out_buffer_length = buffer_length;
  }
  return buffer ? iree_ok_status()
                : iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_format(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_buffer_view_format_impl(
      buffer_view, max_element_count, buffer_capacity, buffer,
      out_buffer_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// TODO(benvanik): streaming all the way down (needs string_util updates).
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_fprint(
    FILE* file, const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(file);
  IREE_ASSERT_ARGUMENT(buffer_view);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Query the string length (in characters).
  iree_host_size_t buffer_length = 0;
  iree_status_t status = iree_hal_buffer_view_format(
      buffer_view, max_element_count, 0, NULL, &buffer_length);
  if (!iree_status_is_out_of_range(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Allocate scratch space to format in to.
  // We should be streaming.
  iree_host_size_t buffer_capacity = buffer_length + 1;  // NUL
  char* buffer = NULL;
  status =
      iree_allocator_malloc(host_allocator, buffer_capacity, (void**)&buffer);

  // Format the buffer into the string storage.
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_view_format(buffer_view, max_element_count,
                                    buffer_capacity, buffer, &buffer_length);
  }

  // Dump to the file.
  if (iree_status_is_ok(status)) {
    fprintf(file, "%.*s", (int)buffer_length, buffer);
  }

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_append_to_builder(
    iree_hal_buffer_view_t* buffer_view, iree_host_size_t max_element_count,
    iree_string_builder_t* builder) {
  // NOTE: we could see how many bytes are available in the builder (capacity -
  // size) and then pass those in to the initial format - if there's enough
  // space it'll fill what it needs. We'd need to adjust the string builder
  // afterward somehow.
  iree_host_size_t required_length = 0;
  iree_status_t status = iree_hal_buffer_view_format(
      buffer_view, max_element_count, /*buffer_capacity=*/0, /*buffer=*/NULL,
      &required_length);
  if (!iree_status_is_out_of_range(status)) return status;
  char* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_inline(builder, required_length, &buffer));
  if (!buffer) return iree_ok_status();
  return iree_hal_buffer_view_format(buffer_view, max_element_count,
                                     required_length + /*NUL=*/1, buffer,
                                     &required_length);
}
