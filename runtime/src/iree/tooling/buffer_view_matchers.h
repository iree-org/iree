// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Buffer view matchers
//===----------------------------------------------------------------------===//
//
// Provides a set of gtest-like buffer views matchers that can either be
// wrapped in C++ and exposed directly to gtest or used programmatically to
// perform buffer view comparisons.
//
// Each matcher has a simple method that returns whether the match was
// successful. Most code should prefer those.
//
// Support for rare element types and encodings are added as-needed and will
// generally return match failure or a status error when unimplemented.
//
// TODO(benvanik): add C++ wrappers in iree/testing/.

#ifndef IREE_TOOLING_BUFFER_VIEW_MATCHERS_H_
#define IREE_TOOLING_BUFFER_VIEW_MATCHERS_H_

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_buffer_equality_t
//===----------------------------------------------------------------------===//

typedef enum {
  // a == b
  IREE_HAL_BUFFER_EQUALITY_EXACT = 0,
  // abs(a - b) <= threshold
  IREE_HAL_BUFFER_EQUALITY_APPROXIMATE_ABSOLUTE,
} iree_hal_buffer_equality_mode_t;

// TODO(benvanik): initializers/configuration for equality comparisons.
typedef struct {
  iree_hal_buffer_equality_mode_t mode;
  // TODO(benvanik): allow override in approximate modes (ULP, abs/rel diff).
  // For now we just have some hardcoded types that are used in place of
  // compile-time constants. Consider these provisional.
  float f16_threshold;
  float f32_threshold;
  double f64_threshold;
} iree_hal_buffer_equality_t;

// Variant type storing known HAL buffer elements.
typedef struct {
  iree_hal_element_type_t type;
  union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
    uint8_t storage[8];  // max size of all value types
  };
} iree_hal_buffer_element_t;

static inline iree_hal_buffer_element_t iree_hal_make_buffer_element_i8(
    int8_t value) {
  iree_hal_buffer_element_t element;
  element.type = IREE_HAL_ELEMENT_TYPE_INT_8;
  element.i8 = value;
  return element;
}

static inline iree_hal_buffer_element_t iree_hal_make_buffer_element_i16(
    int16_t value) {
  iree_hal_buffer_element_t element;
  element.type = IREE_HAL_ELEMENT_TYPE_INT_16;
  element.i16 = value;
  return element;
}

static inline iree_hal_buffer_element_t iree_hal_make_buffer_element_i32(
    int32_t value) {
  iree_hal_buffer_element_t element;
  element.type = IREE_HAL_ELEMENT_TYPE_INT_32;
  element.i32 = value;
  return element;
}

static inline iree_hal_buffer_element_t iree_hal_make_buffer_element_i64(
    int64_t value) {
  iree_hal_buffer_element_t element;
  element.type = IREE_HAL_ELEMENT_TYPE_INT_64;
  element.i64 = value;
  return element;
}

static inline iree_hal_buffer_element_t iree_hal_make_buffer_element_f16(
    float value) {
  iree_hal_buffer_element_t element;
  element.type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  element.i16 = iree_math_f32_to_f16(value);
  return element;
}

static inline iree_hal_buffer_element_t iree_hal_make_buffer_element_f32(
    float value) {
  iree_hal_buffer_element_t element;
  element.type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  element.f32 = value;
  return element;
}

static inline iree_hal_buffer_element_t iree_hal_make_buffer_element_f64(
    double value) {
  iree_hal_buffer_element_t element;
  element.type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
  element.f64 = value;
  return element;
}

// Returns true if all elements match the uniform value based on |equality|.
// |out_index| will contain the first index that does not match.
bool iree_hal_compare_buffer_elements_broadcast(
    iree_hal_buffer_equality_t equality,
    iree_hal_buffer_element_t expected_element, iree_host_size_t element_count,
    iree_const_byte_span_t actual_elements, iree_host_size_t* out_index);

// Returns true if all elements match based on |equality|.
// |out_index| will contain the first index that does not match.
bool iree_hal_compare_buffer_elements_elementwise(
    iree_hal_buffer_equality_t equality, iree_hal_element_type_t element_type,
    iree_host_size_t element_count, iree_const_byte_span_t expected_elements,
    iree_const_byte_span_t actual_elements, iree_host_size_t* out_index);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_metadata_matcher_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_host_size_t shape_rank;
  iree_hal_dim_t shape[128];
  iree_hal_element_type_t element_type;
  iree_hal_encoding_type_t encoding_type;
} iree_hal_buffer_view_metadata_matcher_t;

iree_status_t iree_hal_buffer_view_metadata_matcher_initialize(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_view_metadata_matcher_t* out_matcher);
void iree_hal_buffer_view_metadata_matcher_deinitialize(
    iree_hal_buffer_view_metadata_matcher_t* matcher);
iree_status_t iree_hal_buffer_view_metadata_matcher_describe(
    iree_hal_buffer_view_metadata_matcher_t* matcher,
    iree_string_builder_t* builder);
iree_status_t iree_hal_buffer_view_metadata_matcher_match(
    iree_hal_buffer_view_metadata_matcher_t* matcher,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched);

// Matches |matchee| against the given metadata.
// Use IREE_HAL_ELEMENT_TYPE_NONE to ignore |element_type| and
// use IREE_HAL_ENCODING_TYPE_OPAQUE to ignore |encoding_type|.
iree_status_t iree_hal_buffer_view_match_metadata(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_hal_buffer_view_t* matchee,
    iree_string_builder_t* builder, bool* out_matched);

// Matches |matchee| against |expected| if all metadata (shape, encoding, etc)
// is equivalent.
iree_status_t iree_hal_buffer_view_match_metadata_like(
    iree_hal_buffer_view_t* expected, iree_hal_buffer_view_t* matchee,
    iree_string_builder_t* builder, bool* out_matched);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_element_matcher_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_buffer_equality_t equality;
  iree_hal_buffer_element_t value;
} iree_hal_buffer_view_element_matcher_t;

iree_status_t iree_hal_buffer_view_element_matcher_initialize(
    iree_hal_buffer_equality_t equality, iree_hal_buffer_element_t value,
    iree_hal_buffer_view_element_matcher_t* out_matcher);
void iree_hal_buffer_view_element_matcher_deinitialize(
    iree_hal_buffer_view_element_matcher_t* matcher);
iree_status_t iree_hal_buffer_view_element_matcher_describe(
    iree_hal_buffer_view_element_matcher_t* matcher,
    iree_string_builder_t* builder);
iree_status_t iree_hal_buffer_view_element_matcher_match(
    iree_hal_buffer_view_element_matcher_t* matcher,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched);

// Matches all elements of |matchee| against |value|.
iree_status_t iree_hal_buffer_view_match_elements(
    iree_hal_buffer_equality_t equality, iree_hal_buffer_element_t value,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_array_matcher_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_buffer_equality_t equality;
  iree_hal_element_type_t element_type;
  iree_host_size_t element_count;
  // TODO(benvanik): copy in? would make easier to take from std::vector.
  iree_const_byte_span_t elements;  // unowned
} iree_hal_buffer_view_array_matcher_t;

iree_status_t iree_hal_buffer_view_array_matcher_initialize(
    iree_hal_buffer_equality_t equality, iree_hal_element_type_t element_type,
    iree_host_size_t element_count, iree_const_byte_span_t elements,
    iree_hal_buffer_view_array_matcher_t* out_matcher);
void iree_hal_buffer_view_array_matcher_deinitialize(
    iree_hal_buffer_view_array_matcher_t* matcher);
iree_status_t iree_hal_buffer_view_array_matcher_describe(
    iree_hal_buffer_view_array_matcher_t* matcher,
    iree_string_builder_t* builder);
iree_status_t iree_hal_buffer_view_array_matcher_match(
    iree_hal_buffer_view_array_matcher_t* matcher,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched);

// Matches |matchee| against all |element_count| elements in |elements|.
// The element count of |matchee| must be equal to |element_count|.
iree_status_t iree_hal_buffer_view_match_array(
    iree_hal_buffer_equality_t equality, iree_hal_element_type_t element_type,
    iree_host_size_t element_count, iree_const_byte_span_t elements,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_matcher_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_buffer_equality_t equality;
  iree_hal_buffer_view_t* expected;
} iree_hal_buffer_view_matcher_t;

iree_status_t iree_hal_buffer_view_matcher_initialize(
    iree_hal_buffer_equality_t equality, iree_hal_buffer_view_t* expected,
    iree_hal_buffer_view_matcher_t* out_matcher);
void iree_hal_buffer_view_matcher_deinitialize(
    iree_hal_buffer_view_matcher_t* matcher);
iree_status_t iree_hal_buffer_view_matcher_describe(
    iree_hal_buffer_view_matcher_t* matcher, iree_string_builder_t* builder);
iree_status_t iree_hal_buffer_view_matcher_match(
    iree_hal_buffer_view_matcher_t* matcher, iree_hal_buffer_view_t* matchee,
    iree_string_builder_t* builder, bool* out_matched);

// Matches |matchee| against |expected| for both metadata and elements.
iree_status_t iree_hal_buffer_view_match_equal(
    iree_hal_buffer_equality_t equality, iree_hal_buffer_view_t* expected,
    iree_hal_buffer_view_t* matchee, iree_string_builder_t* builder,
    bool* out_matched);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_BUFFER_VIEW_MATCHERS_H_
