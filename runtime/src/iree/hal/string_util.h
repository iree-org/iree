// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STRING_UTIL_H_
#define IREE_HAL_STRING_UTIL_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_view.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parses a serialized set of shape dimensions using the canonical shape format
// (the same as produced by iree_hal_format_shape).
IREE_API_EXPORT iree_status_t iree_hal_parse_shape(
    iree_string_view_t value, iree_host_size_t shape_capacity,
    iree_host_size_t* out_shape_rank, iree_hal_dim_t* out_shape);

// Converts shape dimensions into a `4x5x6` format.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t
iree_hal_format_shape(iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
                      iree_host_size_t buffer_capacity, char* buffer,
                      iree_host_size_t* out_buffer_length);

// Appends shape dimensions to |string_builder| in a `4x5x6` format.
IREE_API_EXPORT iree_status_t iree_hal_append_shape_string(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_string_builder_t* string_builder);

// Parses a serialized iree_hal_element_type_t and sets |out_element_type| if
// it is valid. The format is the same as produced by
// iree_hal_format_element_type.
IREE_API_EXPORT iree_status_t iree_hal_parse_element_type(
    iree_string_view_t value, iree_hal_element_type_t* out_element_type);

// Converts an iree_hal_element_type_t enum value to a canonical string
// representation, like `IREE_HAL_ELEMENT_TYPE_FLOAT_16` to `f16`.
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t iree_hal_format_element_type(
    iree_hal_element_type_t element_type, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

// Appends an element type to |string_builder| such as `f16`.
IREE_API_EXPORT iree_status_t
iree_hal_append_element_type_string(iree_hal_element_type_t element_type,
                                    iree_string_builder_t* string_builder);

// Parses a shape and type from a `[shape]x[type]` string |value|.
// Behaves the same as calling iree_hal_parse_shape and
// iree_hal_parse_element_type. Ignores any trailing `=`.
IREE_API_EXPORT iree_status_t iree_hal_parse_shape_and_element_type(
    iree_string_view_t value, iree_host_size_t shape_capacity,
    iree_host_size_t* out_shape_rank, iree_hal_dim_t* out_shape,
    iree_hal_element_type_t* out_element_type);

// Appends shape dimensions and element type to |string_builder| as `4x5xf32`.
IREE_API_EXPORT iree_status_t iree_hal_append_shape_and_element_type_string(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_string_builder_t* string_builder);

// Parses a serialized element of |element_type| to its in-memory form.
// |data_ptr| must be at least large enough to contain the bytes of the element.
// For example, "1.2" of type IREE_HAL_ELEMENT_TYPE_FLOAT32 will write the 4
// byte float value of 1.2 to |data_ptr|.
IREE_API_EXPORT iree_status_t iree_hal_parse_element(
    iree_string_view_t data_str, iree_hal_element_type_t element_type,
    iree_byte_span_t data_ptr);

// Converts a single element of |element_type| to a string.
//
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters. Returns
// IREE_STATUS_OUT_OF_RANGE if the buffer capacity is insufficient to hold the
// formatted elements and |out_buffer_length| will contain the required size.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t iree_hal_format_element(
    iree_const_byte_span_t data, iree_hal_element_type_t element_type,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length);

// Parses a serialized set of elements of the given |element_type|.
// The resulting parsed data is written to |data_ptr|, which must be at least
// large enough to contain the parsed elements. The format is the same as
// produced by iree_hal_format_buffer_elements. Supports additional inputs of
// empty to denote a 0 fill and a single element to denote a splat.
IREE_API_EXPORT iree_status_t iree_hal_parse_buffer_elements(
    iree_string_view_t data_str, iree_hal_element_type_t element_type,
    iree_byte_span_t data_ptr);

// Converts a shaped buffer of |element_type| elements to a string.
// This will include []'s to denote each dimension, for example for a shape of
// 2x3 the elements will be formatted as `[1 2 3][4 5 6]`.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
//
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters. Returns
// IREE_STATUS_OUT_OF_RANGE if the buffer capacity is insufficient to hold the
// formatted elements and |out_buffer_length| will contain the required size.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t iree_hal_format_buffer_elements(
    iree_const_byte_span_t data, iree_host_size_t shape_rank,
    const iree_hal_dim_t* shape, iree_hal_element_type_t element_type,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STRING_UTIL_H_
