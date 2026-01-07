// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Zero-copy JSON parsing utilities.
//
// This is a minimal JSON parser designed for parsing configuration files and
// metadata. It operates on string views without allocation and returns spans
// into the original input. For string values containing escape sequences, use
// iree_json_unescape_string() to decode them into a caller-provided buffer.
//
// The parser implements RFC 8259 (JSON) with the following constraints:
// - No streaming support (entire input must be in memory)
// - String values are returned raw (with escape sequences)
// - Number parsing is basic (use iree_json_parse_* helpers for conversion)
//
// Usage:
//   iree_string_view_t json = IREE_SV("{\"key\": 123}");
//   iree_string_view_t value;
//   IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(json, IREE_SV("key"),
//                                                      &value));
//   int64_t number;
//   IREE_RETURN_IF_ERROR(iree_json_parse_int64(value, &number));

#ifndef IREE_BASE_INTERNAL_JSON_H_
#define IREE_BASE_INTERNAL_JSON_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Value Consumption
//===----------------------------------------------------------------------===//
// These functions advance |str| past a JSON value and return the consumed
// span in |out_value|. The input string view acts as a cursor that is modified
// in place.

// Consumes any JSON value from |str| and returns it.
// Handles strings, numbers, objects, arrays, true, false, and null.
// Leading whitespace is trimmed. On success, |str| points after the value.
iree_status_t iree_json_consume_value(iree_string_view_t* str,
                                      iree_string_view_t* out_value);

// Consumes the literal |keyword| from |str|.
// Returns the keyword in |out_value| on success.
iree_status_t iree_json_consume_keyword(iree_string_view_t* str,
                                        iree_string_view_t keyword,
                                        iree_string_view_t* out_value);

// Consumes a JSON number from |str|.
// Handles integers, negative numbers, decimals, and scientific notation.
// The returned value is the raw string span; use iree_json_parse_* to convert.
iree_status_t iree_json_consume_number(iree_string_view_t* str,
                                       iree_string_view_t* out_value);

// Consumes a JSON string from |str|.
// The returned value excludes the surrounding quotes but includes any escape
// sequences as-is. Use iree_json_unescape_string() to decode escape sequences.
iree_status_t iree_json_consume_string(iree_string_view_t* str,
                                       iree_string_view_t* out_value);

// Consumes a JSON object from |str|, including all nested content.
// The returned value includes the surrounding braces: {"key": value}
iree_status_t iree_json_consume_object(iree_string_view_t* str,
                                       iree_string_view_t* out_value);

// Consumes a JSON array from |str|, including all nested content.
// The returned value includes the surrounding brackets: [value, ...]
iree_status_t iree_json_consume_array(iree_string_view_t* str,
                                      iree_string_view_t* out_value);

//===----------------------------------------------------------------------===//
// Object Operations
//===----------------------------------------------------------------------===//

// Callback for object enumeration.
// |key| is the raw key string (without quotes, may contain escape sequences).
// |value| is the raw value span (may be any JSON type).
// Return iree_ok_status() to continue, or any error status to abort.
// To stop early without error, use iree_status_from_code(IREE_STATUS_CANCELLED)
// (not iree_make_status which may allocate).
typedef iree_status_t(IREE_API_PTR* iree_json_object_visitor_fn_t)(
    void* user_data, iree_string_view_t key, iree_string_view_t value);

// Enumerates all key-value pairs in a JSON object.
// |object_value| must include the surrounding braces.
// The visitor is called for each key-value pair in order.
// Returning IREE_STATUS_CANCELLED from the visitor stops enumeration early
// (not treated as an error). Use iree_status_from_code(IREE_STATUS_CANCELLED)
// to avoid allocation.
iree_status_t iree_json_enumerate_object(iree_string_view_t object_value,
                                         iree_json_object_visitor_fn_t visitor,
                                         void* user_data);

// Looks up a key in a JSON object and returns its value.
// |object_value| must include the surrounding braces.
// |key| is matched exactly (escape sequences are not compared semantically).
// Returns empty string view if key is not found (not an error).
iree_status_t iree_json_lookup_object_value(iree_string_view_t object_value,
                                            iree_string_view_t key,
                                            iree_string_view_t* out_value);

//===----------------------------------------------------------------------===//
// Array Operations
//===----------------------------------------------------------------------===//

// Callback for array enumeration.
// |index| is the zero-based element index.
// |value| is the raw value span (may be any JSON type).
// Return iree_ok_status() to continue, or any error status to abort.
// To stop early without error, use iree_status_from_code(IREE_STATUS_CANCELLED)
// (not iree_make_status which may allocate).
typedef iree_status_t(IREE_API_PTR* iree_json_array_visitor_fn_t)(
    void* user_data, iree_host_size_t index, iree_string_view_t value);

// Enumerates all elements in a JSON array.
// |array_value| must include the surrounding brackets.
// The visitor is called for each element in order with its index.
// Returning IREE_STATUS_CANCELLED from the visitor stops enumeration early
// (not treated as an error). Use iree_status_from_code(IREE_STATUS_CANCELLED)
// to avoid allocation.
iree_status_t iree_json_enumerate_array(iree_string_view_t array_value,
                                        iree_json_array_visitor_fn_t visitor,
                                        void* user_data);

//===----------------------------------------------------------------------===//
// String Unescaping
//===----------------------------------------------------------------------===//

// Unescapes a JSON string value, writing UTF-8 to the output buffer.
// |escaped_string| is the raw string content (without surrounding quotes).
// |out_string_capacity| is the size of the output buffer in bytes.
// |out_string| receives the unescaped UTF-8 string (not NUL-terminated).
// |out_string_length| receives the actual length written.
//
// If |out_string_capacity| is 0 or |out_string| is NULL, only computes the
// required length and returns it in |out_string_length|.
//
// Handles all RFC 8259 escape sequences:
//   \" \\ \/ \b \f \n \r \t  - single character escapes
//   \uNNNN                   - basic multilingual plane codepoints
//   \uD800-\uDBFF \uDC00-\uDFFF - surrogate pairs for codepoints > U+FFFF
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if buffer is too small, with
// |out_string_length| set to the required size.
// Returns IREE_STATUS_INVALID_ARGUMENT for invalid escape sequences.
iree_status_t iree_json_unescape_string(iree_string_view_t escaped_string,
                                        iree_host_size_t out_string_capacity,
                                        char* out_string,
                                        iree_host_size_t* out_string_length);

//===----------------------------------------------------------------------===//
// Number Parsing Helpers
//===----------------------------------------------------------------------===//

// Parses a JSON number as a signed 64-bit integer.
// Handles negative numbers. Returns error for floats or out-of-range values.
iree_status_t iree_json_parse_int64(iree_string_view_t value, int64_t* out);

// Parses a JSON number as an unsigned 64-bit integer.
// Returns error for negative numbers, floats, or out-of-range values.
iree_status_t iree_json_parse_uint64(iree_string_view_t value, uint64_t* out);

// Parses a JSON number as a double-precision float.
// Handles integers, decimals, and scientific notation.
iree_status_t iree_json_parse_double(iree_string_view_t value, double* out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_JSON_H_
