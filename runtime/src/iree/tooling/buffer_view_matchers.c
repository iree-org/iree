// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/buffer_view_matchers.h"

#include <math.h>

#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_buffer_equality_t
//===----------------------------------------------------------------------===//

static iree_hal_buffer_element_t iree_hal_buffer_element_at(
    iree_hal_element_type_t element_type, iree_const_byte_span_t elements,
    iree_host_size_t index) {
  iree_host_size_t element_size =
      iree_hal_element_dense_byte_count(element_type);
  iree_const_byte_span_t element_data = iree_make_const_byte_span(
      elements.data + index * element_size, element_size);
  iree_hal_buffer_element_t element = {
      .type = element_type,
  };
  memcpy(element.storage, element_data.data, element_size);
  return element;
}

static iree_status_t iree_hal_append_element_string(
    iree_hal_buffer_element_t value, iree_string_builder_t* builder) {
  char temp[64];
  iree_host_size_t temp_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_format_element(
      iree_make_const_byte_span(value.storage,
                                iree_hal_element_dense_byte_count(value.type)),
      value.type, sizeof(temp), temp, &temp_length));
  return iree_string_builder_append_string(
      builder, iree_make_string_view(temp, temp_length));
}

static iree_status_t iree_hal_append_element_mismatch_string(
    iree_host_size_t index, iree_hal_buffer_element_t expected_element,
    iree_hal_buffer_element_t actual_element, iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "element at index %" PRIhsz " (", index));
  IREE_RETURN_IF_ERROR(iree_hal_append_element_string(actual_element, builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV(") does not match the expected (")));
  IREE_RETURN_IF_ERROR(
      iree_hal_append_element_string(expected_element, builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV(")")));
  return iree_ok_status();
}

static bool iree_hal_compare_strided_elements_exact(
    iree_hal_element_type_t element_type, iree_host_size_t element_count,
    iree_const_byte_span_t expected_elements, iree_host_size_t expected_stride,
    iree_const_byte_span_t actual_elements, iree_host_size_t actual_stride,
    iree_host_size_t* out_index) {
  const iree_host_size_t element_size =
      iree_hal_element_dense_byte_count(element_type);
  const uint8_t* expected_ptr = expected_elements.data;
  const uint8_t* actual_ptr = actual_elements.data;
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    int cmp = memcmp(expected_ptr, actual_ptr, element_size);
    if (cmp != 0) {
      *out_index = i;
      return false;
    }
    expected_ptr += expected_stride * element_size;
    actual_ptr += actual_stride * element_size;
  }
  return true;
}

static bool iree_hal_compare_strided_elements_approximate_absolute_f16(
    iree_hal_buffer_equality_t equality, iree_host_size_t element_count,
    const uint16_t* expected_ptr, iree_host_size_t expected_stride,
    const uint16_t* actual_ptr, iree_host_size_t actual_stride,
    iree_host_size_t* out_index) {
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    if (fabsf(iree_math_f16_to_f32(*expected_ptr) -
              iree_math_f16_to_f32(*actual_ptr)) > equality.f16_threshold) {
      *out_index = i;
      return false;
    }
    expected_ptr += expected_stride;
    actual_ptr += actual_stride;
  }
  return true;
}

static bool iree_hal_compare_strided_elements_approximate_absolute_f32(
    iree_hal_buffer_equality_t equality, iree_host_size_t element_count,
    const float* expected_ptr, iree_host_size_t expected_stride,
    const float* actual_ptr, iree_host_size_t actual_stride,
    iree_host_size_t* out_index) {
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    if (fabsf(*expected_ptr - *actual_ptr) > equality.f32_threshold) {
      *out_index = i;
      return false;
    }
    expected_ptr += expected_stride;
    actual_ptr += actual_stride;
  }
  return true;
}

static bool iree_hal_compare_strided_elements_approximate_absolute_f64(
    iree_hal_buffer_equality_t equality, iree_host_size_t element_count,
    const double* expected_ptr, iree_host_size_t expected_stride,
    const double* actual_ptr, iree_host_size_t actual_stride,
    iree_host_size_t* out_index) {
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    if (fabs(*expected_ptr - *actual_ptr) > equality.f64_threshold) {
      *out_index = i;
      return false;
    }
    expected_ptr += expected_stride;
    actual_ptr += actual_stride;
  }
  return true;
}

static bool iree_hal_compare_strided_elements_approximate_absolute(
    iree_hal_buffer_equality_t equality, iree_hal_element_type_t element_type,
    iree_host_size_t element_count, iree_const_byte_span_t expected_elements,
    iree_host_size_t expected_stride, iree_const_byte_span_t actual_elements,
    iree_host_size_t actual_stride, iree_host_size_t* out_index) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return iree_hal_compare_strided_elements_approximate_absolute_f16(
          equality, element_count, (const uint16_t*)expected_elements.data,
          expected_stride, (const uint16_t*)actual_elements.data, actual_stride,
          out_index);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return iree_hal_compare_strided_elements_approximate_absolute_f32(
          equality, element_count, (const float*)expected_elements.data,
          expected_stride, (const float*)actual_elements.data, actual_stride,
          out_index);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return iree_hal_compare_strided_elements_approximate_absolute_f64(
          equality, element_count, (const double*)expected_elements.data,
          expected_stride, (const double*)actual_elements.data, actual_stride,
          out_index);
    default:
      return iree_hal_compare_strided_elements_exact(
          element_type, element_count, expected_elements, expected_stride,
          actual_elements, actual_stride, out_index);
  }
}

// Compares two buffers element by element.
// The provided strides (in elements) are applied up to |element_count|.
static bool iree_hal_compare_strided_elements(
    iree_hal_buffer_equality_t equality, iree_hal_element_type_t element_type,
    iree_host_size_t element_count, iree_const_byte_span_t expected_elements,
    iree_host_size_t expected_stride, iree_const_byte_span_t actual_elements,
    iree_host_size_t actual_stride, iree_host_size_t* out_index) {
  switch (equality.mode) {
    case IREE_HAL_BUFFER_EQUALITY_EXACT:
      return iree_hal_compare_strided_elements_exact(
          element_type, element_count, expected_elements, expected_stride,
          actual_elements, actual_stride, out_index);
    case IREE_HAL_BUFFER_EQUALITY_APPROXIMATE_ABSOLUTE:
      return iree_hal_compare_strided_elements_approximate_absolute(
          equality, element_type, element_count, expected_elements,
          expected_stride, actual_elements, actual_stride, out_index);
    default:
      IREE_ASSERT(false && "unhandled equality mode");
      return false;
  }
}

bool iree_hal_compare_buffer_elements_broadcast(
    iree_hal_buffer_equality_t equality,
    iree_hal_buffer_element_t expected_element, iree_host_size_t element_count,
    iree_const_byte_span_t actual_elements, iree_host_size_t* out_index) {
  return iree_hal_compare_strided_elements(
      equality, expected_element.type, element_count,
      iree_make_const_byte_span(
          expected_element.storage,
          iree_hal_element_dense_byte_count(expected_element.type)),
      /*expected_stride=*/0, actual_elements, /*actual_stride=*/1, out_index);
}

bool iree_hal_compare_buffer_elements_elementwise(
    iree_hal_buffer_equality_t equality, iree_hal_element_type_t element_type,
    iree_host_size_t element_count, iree_const_byte_span_t expected_elements,
    iree_const_byte_span_t actual_elements, iree_host_size_t* out_index) {
  return iree_hal_compare_strided_elements(
      equality, element_type, element_count, expected_elements,
      /*expected_stride=*/1, actual_elements, /*actual_stride=*/1, out_index);
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_metadata_matcher_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_buffer_view_metadata_matcher_initialize(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_view_metadata_matcher_t* out_matcher) {
  IREE_ASSERT_ARGUMENT(!shape_rank || shape);
  memset(out_matcher, 0, sizeof(*out_matcher));
  if (shape_rank > IREE_ARRAYSIZE(out_matcher->shape)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "maximum shape rank exceeded");
  }
  out_matcher->shape_rank = shape_rank;
  memcpy(out_matcher->shape, shape, shape_rank * sizeof(*shape));
  out_matcher->element_type = element_type;
  out_matcher->encoding_type = encoding_type;
  return iree_ok_status();
}

void iree_hal_buffer_view_metadata_matcher_deinitialize(
    iree_hal_buffer_view_metadata_matcher_t* matcher) {
  IREE_ASSERT_ARGUMENT(matcher);
  memset(matcher, 0, sizeof(*matcher));
}

iree_status_t iree_hal_buffer_view_metadata_matcher_describe(
    iree_hal_buffer_view_metadata_matcher_t* matcher,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(matcher);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("matches ")));
  IREE_RETURN_IF_ERROR(iree_hal_append_shape_and_element_type_string(
      matcher->shape_rank, matcher->shape, matcher->element_type, builder));
  return iree_ok_status();
}

static bool iree_hal_buffer_view_shape_matches(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_buffer_view_t* matchee) {
  if (shape_rank != iree_hal_buffer_view_shape_rank(matchee)) return false;
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    if (shape[i] != iree_hal_buffer_view_shape_dim(matchee, i)) return false;
  }
  return true;
}

iree_status_t iree_hal_buffer_view_metadata_matcher_match(
    iree_hal_buffer_view_metadata_matcher_t* matcher,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched) {
  IREE_ASSERT_ARGUMENT(matcher);
  IREE_ASSERT_ARGUMENT(matchee);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_matched);
  *out_matched = false;

  const bool shape_match = iree_hal_buffer_view_shape_matches(
      matcher->shape_rank, matcher->shape, matchee);
  const bool element_type_match =
      matcher->element_type == IREE_HAL_ELEMENT_TYPE_NONE ||
      matcher->element_type == iree_hal_buffer_view_element_type(matchee);
  const bool encoding_type_match =
      matcher->encoding_type == IREE_HAL_ENCODING_TYPE_OPAQUE ||
      matcher->encoding_type == iree_hal_buffer_view_encoding_type(matchee);
  if (shape_match && element_type_match && encoding_type_match) {
    *out_matched = true;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("metadata is ")));
  IREE_RETURN_IF_ERROR(iree_hal_append_shape_and_element_type_string(
      iree_hal_buffer_view_shape_rank(matchee),
      iree_hal_buffer_view_shape_dims(matchee),
      iree_hal_buffer_view_element_type(matchee), builder));

  *out_matched = false;
  return iree_ok_status();
}

iree_status_t iree_hal_buffer_view_match_metadata(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_hal_buffer_view_t* matchee,
    iree_string_builder_t* builder, bool* out_matched) {
  iree_hal_buffer_view_metadata_matcher_t matcher;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_metadata_matcher_initialize(
      shape_rank, shape, element_type, encoding_type, &matcher));
  iree_status_t status = iree_hal_buffer_view_metadata_matcher_match(
      &matcher, matchee, builder, out_matched);
  if (iree_status_is_ok(status) && !*out_matched) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("; expected that the view ")));
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_metadata_matcher_describe(&matcher, builder));
  }
  iree_hal_buffer_view_metadata_matcher_deinitialize(&matcher);
  return status;
}

iree_status_t iree_hal_buffer_view_match_metadata_like(
    iree_hal_buffer_view_t* expected, iree_hal_buffer_view_t* matchee,
    iree_string_builder_t* builder, bool* out_matched) {
  IREE_ASSERT_ARGUMENT(expected);
  return iree_hal_buffer_view_match_metadata(
      iree_hal_buffer_view_shape_rank(expected),
      iree_hal_buffer_view_shape_dims(expected),
      iree_hal_buffer_view_element_type(expected),
      iree_hal_buffer_view_encoding_type(expected), matchee, builder,
      out_matched);
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_element_matcher_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_buffer_view_element_matcher_initialize(
    iree_hal_buffer_equality_t equality, iree_hal_buffer_element_t value,
    iree_hal_buffer_view_element_matcher_t* out_matcher) {
  memset(out_matcher, 0, sizeof(*out_matcher));
  out_matcher->equality = equality;
  out_matcher->value = value;
  return iree_ok_status();
}

void iree_hal_buffer_view_element_matcher_deinitialize(
    iree_hal_buffer_view_element_matcher_t* matcher) {
  IREE_ASSERT_ARGUMENT(matcher);
  memset(matcher, 0, sizeof(*matcher));
}

iree_status_t iree_hal_buffer_view_element_matcher_describe(
    iree_hal_buffer_view_element_matcher_t* matcher,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(matcher);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV("has all elements match ")));
  IREE_RETURN_IF_ERROR(
      iree_hal_append_element_type_string(matcher->value.type, builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("=")));
  IREE_RETURN_IF_ERROR(iree_hal_append_element_string(matcher->value, builder));
  return iree_ok_status();
}

iree_status_t iree_hal_buffer_view_element_matcher_match(
    iree_hal_buffer_view_element_matcher_t* matcher,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched) {
  IREE_ASSERT_ARGUMENT(matcher);
  IREE_ASSERT_ARGUMENT(matchee);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_matched);
  *out_matched = false;

  if (iree_hal_buffer_view_encoding_type(matchee) !=
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-dense encodings not supported for matching");
  } else if (iree_hal_buffer_view_element_type(matchee) !=
             matcher->value.type) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("whose element type (")));
    IREE_RETURN_IF_ERROR(iree_hal_append_element_type_string(
        iree_hal_buffer_view_element_type(matchee), builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV(") does not match expected (")));
    IREE_RETURN_IF_ERROR(
        iree_hal_append_element_type_string(matcher->value.type, builder));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV(")")));
    *out_matched = false;
    return iree_ok_status();
  }

  iree_hal_buffer_mapping_t actual_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(matchee), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_WHOLE_BUFFER, &actual_mapping));
  iree_const_byte_span_t actual_contents = iree_make_const_byte_span(
      actual_mapping.contents.data, actual_mapping.contents.data_length);

  iree_host_size_t i = 0;
  const bool all_match = iree_hal_compare_buffer_elements_broadcast(
      matcher->equality, matcher->value,
      iree_hal_buffer_view_element_count(matchee), actual_contents, &i);
  iree_hal_buffer_element_t actual_element = iree_hal_buffer_element_at(
      iree_hal_buffer_view_element_type(matchee), actual_contents, i);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&actual_mapping));

  if (!all_match) {
    IREE_RETURN_IF_ERROR(iree_hal_append_element_mismatch_string(
        i, matcher->value, actual_element, builder));
  }

  *out_matched = all_match;
  return iree_ok_status();
}

iree_status_t iree_hal_buffer_view_match_elements(
    iree_hal_buffer_equality_t equality, iree_hal_buffer_element_t value,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched) {
  iree_hal_buffer_view_element_matcher_t matcher;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_element_matcher_initialize(
      equality, value, &matcher));
  iree_status_t status = iree_hal_buffer_view_element_matcher_match(
      &matcher, matchee, builder, out_matched);
  if (iree_status_is_ok(status) && !*out_matched) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("; expected that the view ")));
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_element_matcher_describe(&matcher, builder));
  }
  iree_hal_buffer_view_element_matcher_deinitialize(&matcher);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_array_matcher_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_buffer_view_array_matcher_initialize(
    iree_hal_buffer_equality_t equality, iree_hal_element_type_t element_type,
    iree_host_size_t element_count, iree_const_byte_span_t elements,
    iree_hal_buffer_view_array_matcher_t* out_matcher) {
  IREE_ASSERT_ARGUMENT(!element_count ||
                       !iree_const_byte_span_is_empty(elements));
  memset(out_matcher, 0, sizeof(*out_matcher));
  out_matcher->equality = equality;
  out_matcher->element_type = element_type;
  out_matcher->element_count = element_count;
  out_matcher->elements = elements;
  return iree_ok_status();
}

void iree_hal_buffer_view_array_matcher_deinitialize(
    iree_hal_buffer_view_array_matcher_t* matcher) {
  IREE_ASSERT_ARGUMENT(matcher);
  memset(matcher, 0, sizeof(*matcher));
}

iree_status_t iree_hal_buffer_view_array_matcher_describe(
    iree_hal_buffer_view_array_matcher_t* matcher,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(matcher);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "has all elements match those in %" PRIhsz " element <array> of ",
      matcher->element_count));
  IREE_RETURN_IF_ERROR(
      iree_hal_append_element_type_string(matcher->element_type, builder));
  // TODO(benvanik): format array contents (elided)? make caller do?
  return iree_ok_status();
}

iree_status_t iree_hal_buffer_view_array_matcher_match(
    iree_hal_buffer_view_array_matcher_t* matcher,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched) {
  IREE_ASSERT_ARGUMENT(matcher);
  IREE_ASSERT_ARGUMENT(matchee);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_matched);
  *out_matched = false;

  if (iree_hal_buffer_view_encoding_type(matchee) !=
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-dense encodings not supported for matching");
  } else if (iree_hal_buffer_view_element_type(matchee) !=
             matcher->element_type) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("whose element type (")));
    IREE_RETURN_IF_ERROR(iree_hal_append_element_type_string(
        iree_hal_buffer_view_element_type(matchee), builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV(") does not match expected (")));
    IREE_RETURN_IF_ERROR(
        iree_hal_append_element_type_string(matcher->element_type, builder));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV(")")));
    *out_matched = false;
    return iree_ok_status();
  } else if (iree_hal_buffer_view_element_count(matchee) !=
             matcher->element_count) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "whose element count (%" PRIhsz ") does not match expected (%" PRIhsz
        ")",
        iree_hal_buffer_view_element_count(matchee), matcher->element_count));
    *out_matched = false;
    return iree_ok_status();
  }

  iree_hal_buffer_mapping_t actual_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(matchee), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_WHOLE_BUFFER, &actual_mapping));
  iree_const_byte_span_t actual_contents = iree_make_const_byte_span(
      actual_mapping.contents.data, actual_mapping.contents.data_length);

  iree_host_size_t i = 0;
  const bool all_match = iree_hal_compare_buffer_elements_elementwise(
      matcher->equality, iree_hal_buffer_view_element_type(matchee),
      iree_hal_buffer_view_element_count(matchee), matcher->elements,
      actual_contents, &i);
  iree_hal_buffer_element_t actual_element = iree_hal_buffer_element_at(
      iree_hal_buffer_view_element_type(matchee), actual_contents, i);
  iree_hal_buffer_element_t expected_element = iree_hal_buffer_element_at(
      iree_hal_buffer_view_element_type(matchee), matcher->elements, i);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&actual_mapping));

  if (!all_match) {
    IREE_RETURN_IF_ERROR(iree_hal_append_element_mismatch_string(
        i, expected_element, actual_element, builder));
  }

  *out_matched = all_match;
  return iree_ok_status();
}

iree_status_t iree_hal_buffer_view_match_array(
    iree_hal_buffer_equality_t equality, iree_hal_element_type_t element_type,
    iree_host_size_t element_count, iree_const_byte_span_t elements,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched) {
  iree_hal_buffer_view_array_matcher_t matcher;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_array_matcher_initialize(
      equality, element_type, element_count, elements, &matcher));
  iree_status_t status = iree_hal_buffer_view_array_matcher_match(
      &matcher, matchee, builder, out_matched);
  if (iree_status_is_ok(status) && !*out_matched) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("; expected that the view ")));
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_array_matcher_describe(&matcher, builder));
  }
  iree_hal_buffer_view_array_matcher_deinitialize(&matcher);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_matcher_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_buffer_view_matcher_initialize(
    iree_hal_buffer_equality_t equality, iree_hal_buffer_view_t* expected,
    iree_hal_buffer_view_matcher_t* out_matcher) {
  IREE_ASSERT_ARGUMENT(expected);
  memset(out_matcher, 0, sizeof(*out_matcher));
  out_matcher->equality = equality;
  out_matcher->expected = expected;
  iree_hal_buffer_view_retain(expected);
  return iree_ok_status();
}

void iree_hal_buffer_view_matcher_deinitialize(
    iree_hal_buffer_view_matcher_t* matcher) {
  IREE_ASSERT_ARGUMENT(matcher);
  iree_hal_buffer_view_release(matcher->expected);
  memset(matcher, 0, sizeof(*matcher));
}

iree_status_t iree_hal_buffer_view_matcher_describe(
    iree_hal_buffer_view_matcher_t* matcher, iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(matcher);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV("is equal to contents of a view of ")));
  IREE_RETURN_IF_ERROR(iree_hal_append_shape_and_element_type_string(
      iree_hal_buffer_view_shape_rank(matcher->expected),
      iree_hal_buffer_view_shape_dims(matcher->expected),
      iree_hal_buffer_view_element_type(matcher->expected), builder));
  // TODO(benvanik): format buffer view contents (elided)? make caller do?
  return iree_ok_status();
}

iree_status_t iree_hal_buffer_view_matcher_match(
    iree_hal_buffer_view_matcher_t* matcher, iree_hal_buffer_view_t* matchee,
    iree_string_builder_t* builder, bool* out_matched) {
  IREE_ASSERT_ARGUMENT(matcher);
  IREE_ASSERT_ARGUMENT(matchee);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_matched);
  *out_matched = false;

  if (iree_hal_buffer_view_encoding_type(matchee) !=
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-dense encodings not supported for matching");
  }

  // Reuse metadata matching to ensure the buffer views are the same shape/type.
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_match_metadata_like(
      matcher->expected, matchee, builder, out_matched));
  if (!*out_matched) return iree_ok_status();

  iree_hal_buffer_mapping_t actual_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(matchee), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_WHOLE_BUFFER, &actual_mapping));
  iree_hal_buffer_mapping_t expected_mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(matcher->expected),
      IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
      IREE_WHOLE_BUFFER, &expected_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_unmap_range(&actual_mapping);
    return status;
  }
  iree_const_byte_span_t actual_contents = iree_make_const_byte_span(
      actual_mapping.contents.data, actual_mapping.contents.data_length);
  iree_const_byte_span_t expected_contents = iree_make_const_byte_span(
      expected_mapping.contents.data, expected_mapping.contents.data_length);

  iree_host_size_t i = 0;
  const bool all_match = iree_hal_compare_buffer_elements_elementwise(
      matcher->equality, iree_hal_buffer_view_element_type(matchee),
      iree_hal_buffer_view_element_count(matchee), expected_contents,
      actual_contents, &i);
  iree_hal_buffer_element_t actual_element = iree_hal_buffer_element_at(
      iree_hal_buffer_view_element_type(matchee), actual_contents, i);
  iree_hal_buffer_element_t expected_element = iree_hal_buffer_element_at(
      iree_hal_buffer_view_element_type(matchee), expected_contents, i);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&actual_mapping));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&expected_mapping));

  if (!all_match) {
    IREE_RETURN_IF_ERROR(iree_hal_append_element_mismatch_string(
        i, expected_element, actual_element, builder));
  }

  *out_matched = all_match;
  return iree_ok_status();
}

iree_status_t iree_hal_buffer_view_match_equal(
    iree_hal_buffer_equality_t equality, iree_hal_buffer_view_t* expected,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched) {
  iree_hal_buffer_view_matcher_t matcher;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_matcher_initialize(equality, expected, &matcher));
  iree_status_t status = iree_hal_buffer_view_matcher_match(
      &matcher, matchee, builder, out_matched);
  if (iree_status_is_ok(status) && !*out_matched) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("; expected that the view ")));
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_matcher_describe(&matcher, builder));
  }
  iree_hal_buffer_view_matcher_deinitialize(&matcher);
  return status;
}
