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

#include "iree/hal/buffer_view.h"

#include <cctype>
#include <cinttypes>
#include <cstdio>

#include "absl/container/inlined_vector.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/string_util.h"

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_parse(
    iree_string_view_t value, iree_hal_allocator_t* buffer_allocator,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_parse");
  IREE_ASSERT_ARGUMENT(buffer_allocator);

  // Strip whitespace that may come along (linefeeds/etc).
  auto string_view =
      absl::StripAsciiWhitespace(absl::string_view(value.data, value.size));
  string_view = absl::StripPrefix(string_view, "\"");
  string_view = absl::StripSuffix(string_view, "\"");
  if (string_view.empty()) {
    // Empty lines are invalid; need at least the shape/type information.
    *out_buffer_view = nullptr;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty string input");
  }

  // The part of the string corresponding to the shape, e.g. 1x2x3.
  absl::string_view shape_str;
  // The part of the string corresponding to the type, e.g. f32
  absl::string_view type_str;
  // The part of the string corresponding to the buffer data, e.g. 1 2 3 4 5 6
  absl::string_view data_str;

  absl::string_view shape_and_type_str;
  auto equal_index = string_view.find('=');
  if (equal_index == std::string::npos) {
    // Treat a lack of = as defaulting the data to zeros.
    shape_and_type_str = string_view;
  } else {
    shape_and_type_str = string_view.substr(0, equal_index);
    data_str = string_view.substr(equal_index + 1);
  }
  auto last_x_index = shape_and_type_str.rfind('x');
  if (last_x_index == std::string::npos) {
    // Scalar.
    type_str = shape_and_type_str;
  } else {
    // Has a shape.
    shape_str = shape_and_type_str.substr(0, last_x_index);
    type_str = shape_and_type_str.substr(last_x_index + 1);
  }

  // AxBxC...
  absl::InlinedVector<iree_hal_dim_t, 6> shape(6);
  iree_host_size_t shape_rank = 0;
  iree_status_t shape_result =
      iree_hal_parse_shape({shape_str.data(), shape_str.length()}, shape.size(),
                           shape.data(), &shape_rank);
  if (iree_status_is_ok(shape_result)) {
    shape.resize(shape_rank);
  } else if (iree_status_is_out_of_range(shape_result)) {
    shape.resize(shape_rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_parse_shape({shape_str.data(), shape_str.length()},
                             shape.size(), shape.data(), &shape_rank));
  } else {
    return shape_result;
  }

  // f32, i32, etc
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_parse_element_type(
      {type_str.data(), type_str.length()}, &element_type));

  // Allocate the buffer we will parse into from the provided allocator.
  iree_device_size_t buffer_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_compute_view_size(
      shape.data(), shape.size(), element_type, &buffer_length));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      buffer_allocator,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      buffer_length, &buffer));

  iree_status_t status;

  // Parse the elements directly into the buffer.
  iree_hal_buffer_mapping_t buffer_mapping;
  status =
      iree_hal_buffer_map_range(buffer, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0,
                                buffer_length, &buffer_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }
  status =
      iree_hal_parse_buffer_elements({data_str.data(), data_str.length()},
                                     element_type, buffer_mapping.contents);
  iree_hal_buffer_unmap_range(&buffer_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }

  // Wrap and pass ownership of the buffer to the buffer view.
  status = iree_hal_buffer_view_create(buffer, shape.data(), shape.size(),
                                       element_type, out_buffer_view);
  iree_hal_buffer_release(buffer);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_format(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_format");
  IREE_ASSERT_ARGUMENT(buffer_view);
  if (out_buffer_length) {
    *out_buffer_length = 0;
  }
  if (buffer && buffer_capacity) {
    buffer[0] = 0;
  }

  iree_status_t status;
  iree_host_size_t buffer_length = 0;
  auto append_char = [&](char c) {
    if (buffer) {
      if (buffer_length < buffer_capacity - 1) {
        buffer[buffer_length] = c;
        buffer[buffer_length + 1] = '\0';
      } else {
        buffer = nullptr;
      }
    }
    ++buffer_length;
  };

  if (iree_hal_buffer_view_shape_rank(buffer_view) > 0) {
    // Shape: 1x2x3
    iree_host_size_t shape_length = 0;
    status = iree_hal_format_shape(iree_hal_buffer_view_shape_dims(buffer_view),
                                   iree_hal_buffer_view_shape_rank(buffer_view),
                                   buffer ? buffer_capacity - buffer_length : 0,
                                   buffer ? buffer + buffer_length : nullptr,
                                   &shape_length);
    buffer_length += shape_length;
    if (iree_status_is_out_of_range(status)) {
      status = iree_status_ignore(status);
      buffer = nullptr;
    } else if (!iree_status_is_ok(status)) {
      return status;
    }

    // Separator: <shape>x<format>
    append_char('x');
  }

  // Element type: f32
  iree_host_size_t element_type_length = 0;
  status = iree_hal_format_element_type(
      iree_hal_buffer_view_element_type(buffer_view),
      buffer ? buffer_capacity - buffer_length : 0,
      buffer ? buffer + buffer_length : nullptr, &element_type_length);
  buffer_length += element_type_length;
  if (iree_status_is_out_of_range(status)) {
    status = iree_status_ignore(status);
    buffer = nullptr;
  } else if (!iree_status_is_ok(status)) {
    return status;
  }

  // Separator: <meta>=<value>
  append_char('=');

  // Buffer contents: 0 1 2 3 ...
  iree_hal_buffer_mapping_t buffer_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(buffer_view), IREE_HAL_MEMORY_ACCESS_READ, 0,
      IREE_WHOLE_BUFFER, &buffer_mapping));
  iree_host_size_t elements_length = 0;
  status = iree_hal_format_buffer_elements(
      iree_const_byte_span_t{buffer_mapping.contents.data,
                             buffer_mapping.contents.data_length},
      iree_hal_buffer_view_shape_dims(buffer_view),
      iree_hal_buffer_view_shape_rank(buffer_view),
      iree_hal_buffer_view_element_type(buffer_view), max_element_count,
      buffer ? buffer_capacity - buffer_length : 0,
      buffer ? buffer + buffer_length : nullptr, &elements_length);
  buffer_length += elements_length;
  iree_hal_buffer_unmap_range(&buffer_mapping);
  if (iree_status_is_out_of_range(status)) {
    status = iree_status_ignore(status);
    buffer = nullptr;
  } else if (!iree_status_is_ok(status)) {
    return status;
  }

  if (out_buffer_length) {
    *out_buffer_length = buffer_length;
  }
  return buffer ? iree_ok_status()
                : iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
}
