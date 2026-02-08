// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Zero-copy JSON/JSONC parsing utilities.
//
// This is a minimal JSON parser designed for parsing configuration files and
// metadata. It operates on string views without allocation and returns spans
// into the original input. For string values containing escape sequences, use
// iree_json_unescape_string() to decode them into a caller-provided buffer.
//
// The parser accepts both RFC 8259 (JSON) and JSONC (JSON with Comments):
// - Single-line comments: // comment to end of line
// - Multi-line comments: /* comment block */ (may span lines in JSONL)
// - Trailing commas: [1, 2,] and {"a": 1,}
// - UTF-8 BOM at start of input (automatically skipped)
//
// Safety:
// - Maximum nesting depth of 128 levels (IREE_JSON_MAX_DEPTH) to prevent DoS
// - Override with -DIREE_JSON_MAX_DEPTH=N at compile time if needed
//
// Constraints:
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
// JSON Value Types
//===----------------------------------------------------------------------===//
// Value types are inferred from the first character of raw JSON text.
// These types allow callbacks to distinguish between JSON constructs without
// inspecting value content (which may be ambiguous after quote stripping).

typedef enum iree_json_value_type_e {
  IREE_JSON_VALUE_TYPE_STRING = 0,  // "..." - quotes stripped in value
  IREE_JSON_VALUE_TYPE_NUMBER,      // 0-9 or - prefix
  IREE_JSON_VALUE_TYPE_OBJECT,      // {...} - braces included in value
  IREE_JSON_VALUE_TYPE_ARRAY,       // [...] - brackets included in value
  IREE_JSON_VALUE_TYPE_TRUE,        // true
  IREE_JSON_VALUE_TYPE_FALSE,       // false
  IREE_JSON_VALUE_TYPE_NULL,        // null
} iree_json_value_type_t;

// Returns the JSON value type for the given first character.
// Used internally by enumeration functions; exposed for testing.
static inline iree_json_value_type_t iree_json_infer_value_type(char c) {
  switch (c) {
    case '"':
      return IREE_JSON_VALUE_TYPE_STRING;
    case '{':
      return IREE_JSON_VALUE_TYPE_OBJECT;
    case '[':
      return IREE_JSON_VALUE_TYPE_ARRAY;
    case 't':
      return IREE_JSON_VALUE_TYPE_TRUE;
    case 'f':
      return IREE_JSON_VALUE_TYPE_FALSE;
    case 'n':
      return IREE_JSON_VALUE_TYPE_NULL;
    default:
      return IREE_JSON_VALUE_TYPE_NUMBER;
  }
}

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
// |value| is the value (strings without quotes, objects/arrays with
// delimiters).
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
// Returns IREE_STATUS_NOT_FOUND if the key is not present in the object.
iree_status_t iree_json_lookup_object_value(iree_string_view_t object_value,
                                            iree_string_view_t key,
                                            iree_string_view_t* out_value);

// Looks up a key in a JSON object and returns its value, with missing key OK.
// |object_value| must include the surrounding braces.
// |key| is matched exactly (escape sequences are not compared semantically).
// Returns an empty string view in |out_value| if the key is not found (not an
// error). This variant is useful for optional fields where missing is valid.
//
// NOTE: The returned iree_status_t MUST be handled (via IREE_RETURN_IF_ERROR or
// iree_status_ignore) even though missing keys are not errors. Ignoring the
// return value leaks status storage on parse errors.
iree_status_t iree_json_try_lookup_object_value(iree_string_view_t object_value,
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

// Extended callback for array enumeration that includes value type.
// |index| is the zero-based element index.
// |type| is the JSON type of the value (inferred before quote stripping).
// |value| is the raw value span (strings without quotes, others with
// delimiters).
// Return iree_ok_status() to continue, or any error status to abort.
// To stop early without error, use iree_status_from_code(IREE_STATUS_CANCELLED)
// (not iree_make_status which may allocate).
typedef iree_status_t(IREE_API_PTR* iree_json_array_visitor_typed_fn_t)(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value);

// Enumerates all elements in a JSON array with type information.
// |array_value| must include the surrounding brackets.
// The visitor is called for each element with its index and inferred type.
// The type is determined from the first character of raw JSON before any
// processing (e.g., before quotes are stripped from strings).
// Returning IREE_STATUS_CANCELLED from the visitor stops enumeration early
// (not treated as an error). Use iree_status_from_code(IREE_STATUS_CANCELLED)
// to avoid allocation.
iree_status_t iree_json_enumerate_array_typed(
    iree_string_view_t array_value, iree_json_array_visitor_typed_fn_t visitor,
    void* user_data);

// Returns the number of elements in a JSON array.
// |array_value| must include the surrounding brackets.
iree_status_t iree_json_array_length(iree_string_view_t array_value,
                                     iree_host_size_t* out_length);

// Gets the element at |index| from a JSON array.
// |array_value| must include the surrounding brackets.
// Returns IREE_STATUS_OUT_OF_RANGE if |index| is beyond the array bounds.
iree_status_t iree_json_array_get(iree_string_view_t array_value,
                                  iree_host_size_t index,
                                  iree_string_view_t* out_value);

//===----------------------------------------------------------------------===//
// JSONL (JSON Lines) Operations
//===----------------------------------------------------------------------===//
// JSONL is a newline-delimited format where each line is a valid JSON value.
// See https://jsonlines.org/ for the specification.

// 1-based line number for JSONL error reporting.
typedef iree_host_size_t iree_json_line_number_t;

// Callback for JSONL line enumeration.
// |line_number| is the 1-based physical line number (for error messages).
// |index| is the 0-based entry index (counting only non-empty lines).
// |value| is the raw value span (may be any JSON type).
// Return iree_ok_status() to continue, or any error status to abort.
// To stop early without error, use iree_status_from_code(IREE_STATUS_CANCELLED)
// (not iree_make_status which may allocate).
typedef iree_status_t(IREE_API_PTR* iree_json_line_visitor_fn_t)(
    void* user_data, iree_json_line_number_t line_number,
    iree_host_size_t index, iree_string_view_t value);

// Enumerates JSON values in a JSONL (newline-delimited JSON) input.
// Each non-empty line is parsed as a complete JSON value.
// Empty lines and lines containing only whitespace are skipped.
// The visitor receives both the 1-based physical line number (for error
// reporting) and the 0-based entry index (for array-like access).
// Returning IREE_STATUS_CANCELLED from the visitor stops enumeration early
// (not treated as an error). Use iree_status_from_code(IREE_STATUS_CANCELLED)
// to avoid allocation.
iree_status_t iree_json_enumerate_lines(iree_string_view_t input,
                                        iree_json_line_visitor_fn_t visitor,
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

// Parses a JSON boolean ("true" or "false") to a bool.
// Returns error for any other value including "null".
iree_status_t iree_json_parse_bool(iree_string_view_t value, bool* out);

// Looks up a required boolean field.
// Returns error if the key is not found, the value is "null", or the value is
// not a valid boolean ("true" or "false").
iree_status_t iree_json_lookup_bool(iree_string_view_t object_value,
                                    iree_string_view_t key, bool* out);

// Looks up a required string field, unescapes it, and writes to |buffer|.
// Returns error if the key is not found, the value is "null", or the buffer is
// too small for the unescaped content.
//
// |buffer| provides the output storage (data and capacity via size).
// |out_length| receives the actual length written (not NUL-terminated).
iree_status_t iree_json_lookup_string(iree_string_view_t object_value,
                                      iree_string_view_t key,
                                      iree_mutable_string_view_t buffer,
                                      iree_host_size_t* out_length);

// Looks up an optional boolean field with a default value.
// Returns |default_value| if the key is not found or the value is "null".
// Returns error if the value is present but not a valid boolean.
iree_status_t iree_json_try_lookup_bool(iree_string_view_t object_value,
                                        iree_string_view_t key,
                                        bool default_value, bool* out);

// Looks up an optional int64 field with a default value.
// Returns |default_value| if the key is not found or the value is "null".
// Returns error if the value is present but not a valid integer.
iree_status_t iree_json_try_lookup_int64(iree_string_view_t object_value,
                                         iree_string_view_t key,
                                         int64_t default_value, int64_t* out);

// Looks up an optional string field, unescapes it, and writes to |buffer|.
// Returns |default_value| copied to buffer if key is not found or value is
// "null". Returns error if value is present but not a string, or if the buffer
// is too small for the unescaped content.
//
// |buffer| provides the output storage (data and capacity via size).
// |out_length| receives the actual length written (not NUL-terminated).
iree_status_t iree_json_try_lookup_string(iree_string_view_t object_value,
                                          iree_string_view_t key,
                                          iree_string_view_t default_value,
                                          iree_mutable_string_view_t buffer,
                                          iree_host_size_t* out_length);

//===----------------------------------------------------------------------===//
// Object Key Validation
//===----------------------------------------------------------------------===//

// Validates that a JSON object contains only allowed keys.
// Returns error listing unknown keys if any are found.
// Example error: "unknown keys in object: foo, bar (supported: type, vocab)"
//
// |object_value| must include the surrounding braces: {"key": value, ...}
// |allowed_keys| is an array of allowed key names.
// |allowed_key_count| is the number of elements in the allowed_keys array.
//
// Keys are matched exactly (escape sequences are not compared semantically).
iree_status_t iree_json_validate_object_keys(
    iree_string_view_t object_value, const iree_string_view_t* allowed_keys,
    iree_host_size_t allowed_key_count);

//===----------------------------------------------------------------------===//
// Unicode Codepoint Parsing
//===----------------------------------------------------------------------===//

// Parses a single Unicode codepoint from a JSON string value.
// First unescapes the string to handle \uNNNN sequences, then decodes the first
// UTF-8 codepoint from the result.
// Returns |default_codepoint| if |value| is empty or results in empty string.
// Returns error if the string contains invalid UTF-8 or is too long.
iree_status_t iree_json_parse_codepoint(iree_string_view_t value,
                                        uint32_t default_codepoint,
                                        uint32_t* out_codepoint);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_JSON_H_
