// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/json.h"

#include <stdlib.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Value Consumption
//===----------------------------------------------------------------------===//

// Forward declaration for mutual recursion.
static iree_status_t iree_json_consume_value_impl(iree_string_view_t* str,
                                                  iree_string_view_t* out_value,
                                                  iree_host_size_t depth);

// Maximum nesting depth for JSON structures (objects/arrays).
// This prevents stack overflow from adversarial inputs.
#ifndef IREE_JSON_MAX_DEPTH
#define IREE_JSON_MAX_DEPTH 128
#endif

// Skips UTF-8 BOM (Byte Order Mark) if present at the start of input.
// BOM is 0xEF 0xBB 0xBF and is commonly added by Windows text editors.
static void iree_json_skip_bom(iree_string_view_t* str) {
  if (str->size >= 3 && (uint8_t)str->data[0] == 0xEF &&
      (uint8_t)str->data[1] == 0xBB && (uint8_t)str->data[2] == 0xBF) {
    *str = iree_string_view_substr(*str, 3, IREE_HOST_SIZE_MAX);
  }
}

// Returns true if |c| is JSON whitespace (space, tab, newline, carriage
// return). This is faster than isspace() which is locale-aware and has call
// overhead.
static inline bool iree_json_is_whitespace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Returns true if |c| is an ASCII digit ('0'-'9').
// This is faster than isdigit() which may have function call overhead.
static inline bool iree_json_is_digit(char c) { return c >= '0' && c <= '9'; }

// Skips leading JSON whitespace characters in place.
// Unlike iree_string_view_trim, this only trims the front (not trailing) and
// uses simple comparisons instead of locale-aware isspace().
static inline void iree_json_skip_whitespace(iree_string_view_t* str) {
  const char* data = str->data;
  iree_host_size_t size = str->size;
  while (size > 0 && iree_json_is_whitespace(*data)) {
    data++;
    size--;
  }
  str->data = data;
  str->size = size;
}

// Skips whitespace and JSONC-style comments (// and /* */).
// This allows parsing both strict JSON and JSONC (JSON with Comments).
static iree_status_t iree_json_skip_whitespace_and_comments(
    iree_string_view_t* str) {
  while (str->size > 0) {
    // Skip leading whitespace.
    iree_json_skip_whitespace(str);
    if (str->size == 0) return iree_ok_status();

    // Check for single-line comment: //
    if (str->size >= 2 && str->data[0] == '/' && str->data[1] == '/') {
      // Find end of line.
      iree_host_size_t i = 2;
      while (i < str->size && str->data[i] != '\n') ++i;
      // Skip past newline if present.
      if (i < str->size && str->data[i] == '\n') ++i;
      *str = iree_string_view_substr(*str, i, IREE_HOST_SIZE_MAX);
      continue;
    }

    // Check for multi-line comment: /* */
    if (str->size >= 2 && str->data[0] == '/' && str->data[1] == '*') {
      iree_host_size_t i = 2;
      bool found_end = false;
      while (i + 1 < str->size) {
        if (str->data[i] == '*' && str->data[i + 1] == '/') {
          *str = iree_string_view_substr(*str, i + 2, IREE_HOST_SIZE_MAX);
          found_end = true;
          break;
        }
        ++i;
      }
      if (!found_end) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unterminated /* */ comment");
      }
      continue;
    }

    // No comment found, done skipping.
    break;
  }
  return iree_ok_status();
}

iree_status_t iree_json_consume_keyword(iree_string_view_t* str,
                                        iree_string_view_t keyword,
                                        iree_string_view_t* out_value) {
  if (iree_string_view_consume_prefix(str, keyword)) {
    *out_value = keyword;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid keyword, expected '%.*s'", (int)keyword.size,
                          keyword.data);
}

iree_status_t iree_json_consume_number(iree_string_view_t* str,
                                       iree_string_view_t* out_value) {
  if (str->size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty number");
  }

  iree_host_size_t i = 0;

  // Optional leading minus sign.
  if (str->data[i] == '-') {
    ++i;
    if (i >= str->size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "number with only minus sign");
    }
  }

  // Integer part.
  if (i < str->size && str->data[i] == '0') {
    // Leading zero must be followed by '.' or end of number (no 00, 01, etc).
    ++i;
  } else {
    // Non-zero leading digit followed by more digits.
    if (i >= str->size || !iree_json_is_digit(str->data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid number");
    }
    while (i < str->size && iree_json_is_digit(str->data[i])) {
      ++i;
    }
  }

  // Optional fractional part.
  if (i < str->size && str->data[i] == '.') {
    ++i;
    if (i >= str->size || !iree_json_is_digit(str->data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid fractional part in number");
    }
    while (i < str->size && iree_json_is_digit(str->data[i])) {
      ++i;
    }
  }

  // Optional exponent part.
  if (i < str->size && (str->data[i] == 'e' || str->data[i] == 'E')) {
    ++i;
    // Optional exponent sign.
    if (i < str->size && (str->data[i] == '+' || str->data[i] == '-')) {
      ++i;
    }
    if (i >= str->size || !iree_json_is_digit(str->data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid exponent in number");
    }
    while (i < str->size && iree_json_is_digit(str->data[i])) {
      ++i;
    }
  }

  if (i == 0 || (i == 1 && str->data[0] == '-')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid number");
  }

  *out_value = iree_string_view_substr(*str, 0, i);
  *str = iree_string_view_remove_prefix(*str, i);
  return iree_ok_status();
}

// Returns true if c is a valid hex digit.
static bool iree_json_is_hex_digit(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
         (c >= 'A' && c <= 'F');
}

iree_status_t iree_json_consume_string(iree_string_view_t* str,
                                       iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  if (!iree_string_view_starts_with(*str, IREE_SV("\""))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing string \" prefix");
  }
  iree_host_size_t start = 1;
  iree_host_size_t end = 0;
  for (iree_host_size_t i = start; i < str->size; ++i) {
    char c = str->data[i];
    if (c == '\"') {
      // Unescaped quote is end of string.
      end = i;
      break;
    } else if ((unsigned char)c < 0x20) {
      // Control characters must be escaped per RFC 8259.
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unescaped control character 0x%02X in string",
                              (unsigned char)c);
    } else if (c == '\\') {
      if (i + 1 >= str->size) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "escape code with no contents");
      }
      // Escaped sequence - usually 1 but may be 4 for \uNNNN.
      switch (str->data[++i]) {
        case '\"':
        case '\\':
        case '/':
        case 'b':
        case 'f':
        case 'n':
        case 'r':
        case 't':
          break;  // ok
        case 'u':
          //   'u' hex hex hex hex
          if (i + 4 >= str->size) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "truncated unicode escape code");
          }
          // Validate hex digits.
          for (int j = 1; j <= 4; ++j) {
            if (!iree_json_is_hex_digit(str->data[i + j])) {
              return iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "invalid hex digit '%c' in unicode escape", str->data[i + j]);
            }
          }
          i += 4;
          break;
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unrecognized string escape code \\%c",
                                  str->data[i]);
      }
    }
  }
  if (end == 0) {
    // Didn't find closing quote.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated string");
  }
  *out_value = iree_string_view_substr(*str, start, end - start);
  *str = iree_string_view_substr(*str, end + 1, IREE_HOST_SIZE_MAX);
  return iree_ok_status();
}

static iree_status_t iree_json_consume_object_impl(
    iree_string_view_t* str, iree_string_view_t* out_value,
    iree_host_size_t depth) {
  if (depth >= IREE_JSON_MAX_DEPTH) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "JSON nesting depth exceeds maximum of %d",
                            (int)IREE_JSON_MAX_DEPTH);
  }
  *out_value = iree_string_view_empty();
  const char* start = str->data;
  if (!iree_string_view_consume_prefix_char(str, '{')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object {");
  }
  IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(str));
  while (!iree_string_view_is_empty(*str)) {
    // Check for end of object.
    if (iree_string_view_starts_with_char(*str, '}')) break;
    // Try to parse key string.
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_string(str, &key));
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(str));
    // Expect : separator.
    if (!iree_string_view_consume_prefix_char(str, ':')) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing object member separator");
    }
    // Scan ahead to get the value span.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value_impl(str, &value, depth + 1));
    // Skip whitespace/comments before comma or closing brace.
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(str));
    // If there's a comma then continue to next member (trailing commas
    // allowed).
    if (!iree_string_view_consume_prefix_char(str, ',')) break;
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(str));
  }
  // Verify closing brace before computing end pointer (str->data may be NULL).
  if (!iree_string_view_starts_with_char(*str, '}')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object }");
  }
  const char* end = str->data + 1;
  iree_string_view_consume_prefix_char(str, '}');
  *out_value = iree_make_string_view(start, end - start);
  return iree_ok_status();
}

iree_status_t iree_json_consume_object(iree_string_view_t* str,
                                       iree_string_view_t* out_value) {
  iree_json_skip_bom(str);
  return iree_json_consume_object_impl(str, out_value, /*depth=*/0);
}

static iree_status_t iree_json_consume_array_impl(iree_string_view_t* str,
                                                  iree_string_view_t* out_value,
                                                  iree_host_size_t depth) {
  if (depth >= IREE_JSON_MAX_DEPTH) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "JSON nesting depth exceeds maximum of %d",
                            (int)IREE_JSON_MAX_DEPTH);
  }
  *out_value = iree_string_view_empty();
  const char* start = str->data;
  if (!iree_string_view_consume_prefix_char(str, '[')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array [");
  }
  IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(str));
  while (!iree_string_view_is_empty(*str)) {
    // Check for end of array.
    if (iree_string_view_starts_with_char(*str, ']')) break;
    // Get the array element.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value_impl(str, &value, depth + 1));
    // Skip whitespace/comments before comma or closing bracket.
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(str));
    // If there's a comma then continue to next element (trailing commas
    // allowed).
    if (!iree_string_view_consume_prefix_char(str, ',')) break;
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(str));
  }
  // Verify closing bracket before computing end pointer (str->data may be
  // NULL).
  if (!iree_string_view_starts_with_char(*str, ']')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array ]");
  }
  const char* end = str->data + 1;
  iree_string_view_consume_prefix_char(str, ']');
  *out_value = iree_make_string_view(start, end - start);
  return iree_ok_status();
}

iree_status_t iree_json_consume_array(iree_string_view_t* str,
                                      iree_string_view_t* out_value) {
  iree_json_skip_bom(str);
  return iree_json_consume_array_impl(str, out_value, /*depth=*/0);
}

static iree_status_t iree_json_consume_value_impl(iree_string_view_t* str,
                                                  iree_string_view_t* out_value,
                                                  iree_host_size_t depth) {
  *out_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(str));
  if (str->size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected JSON value");
  }
  switch (str->data[0]) {
    case '"':
      return iree_json_consume_string(str, out_value);
    case '{':
      return iree_json_consume_object_impl(str, out_value, depth);
    case '[':
      return iree_json_consume_array_impl(str, out_value, depth);
    case 't':
      return iree_json_consume_keyword(str, IREE_SV("true"), out_value);
    case 'f':
      return iree_json_consume_keyword(str, IREE_SV("false"), out_value);
    case 'n':
      return iree_json_consume_keyword(str, IREE_SV("null"), out_value);
    default:
      return iree_json_consume_number(str, out_value);
  }
}

iree_status_t iree_json_consume_value(iree_string_view_t* str,
                                      iree_string_view_t* out_value) {
  iree_json_skip_bom(str);
  return iree_json_consume_value_impl(str, out_value, /*depth=*/0);
}

//===----------------------------------------------------------------------===//
// Object Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_json_enumerate_object(iree_string_view_t object_value,
                                         iree_json_object_visitor_fn_t visitor,
                                         void* user_data) {
  iree_string_view_t str = object_value;
  if (!iree_string_view_consume_prefix_char(&str, '{')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object {");
  }
  IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
  bool cancelled = false;
  while (!iree_string_view_is_empty(str)) {
    // Check for end of object.
    if (iree_string_view_starts_with_char(str, '}')) break;
    // Try to parse key string.
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_string(&str, &key));
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
    // Expect : separator.
    if (!iree_string_view_consume_prefix_char(&str, ':')) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing object member separator");
    }
    // Scan ahead to get the value span.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value(&str, &value));
    // Emit the key-value pair.
    iree_status_t status = visitor(user_data, key, value);
    if (iree_status_is_cancelled(status)) {
      iree_status_ignore(status);
      cancelled = true;
      break;
    }
    IREE_RETURN_IF_ERROR(status);
    // Skip whitespace/comments before comma or closing brace.
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
    // If there's a comma then continue to next member (trailing commas
    // allowed).
    if (!iree_string_view_consume_prefix_char(&str, ',')) break;
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
  }
  // Verify closing brace (unless cancelled early).
  if (!cancelled && !iree_string_view_starts_with_char(str, '}')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object }");
  }
  return iree_ok_status();
}

typedef struct iree_json_lookup_object_value_state_t {
  iree_string_view_t key;
  iree_string_view_t* value;
  bool found;
} iree_json_lookup_object_value_state_t;

static iree_status_t iree_json_lookup_object_value_visitor(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  iree_json_lookup_object_value_state_t* state =
      (iree_json_lookup_object_value_state_t*)user_data;
  if (iree_string_view_equal(key, state->key)) {
    *state->value = value;
    state->found = true;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return iree_ok_status();
}

iree_status_t iree_json_lookup_object_value(iree_string_view_t object_value,
                                            iree_string_view_t key,
                                            iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  iree_json_lookup_object_value_state_t state = {
      .key = key,
      .value = out_value,
      .found = false,
  };
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object(
      object_value, iree_json_lookup_object_value_visitor, &state));
  if (!state.found) {
    return iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }
  return iree_ok_status();
}

iree_status_t iree_json_try_lookup_object_value(iree_string_view_t object_value,
                                                iree_string_view_t key,
                                                iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  iree_json_lookup_object_value_state_t state = {
      .key = key,
      .value = out_value,
      .found = false,
  };
  return iree_json_enumerate_object(
      object_value, iree_json_lookup_object_value_visitor, &state);
}

//===----------------------------------------------------------------------===//
// Array Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_json_enumerate_array(iree_string_view_t array_value,
                                        iree_json_array_visitor_fn_t visitor,
                                        void* user_data) {
  iree_string_view_t str = array_value;
  if (!iree_string_view_consume_prefix_char(&str, '[')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array [");
  }
  IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
  iree_host_size_t index = 0;
  bool cancelled = false;
  while (!iree_string_view_is_empty(str)) {
    // Check for end of array.
    if (iree_string_view_starts_with_char(str, ']')) break;
    // Get the array element.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value(&str, &value));
    // Emit the element.
    iree_status_t status = visitor(user_data, index, value);
    if (iree_status_is_cancelled(status)) {
      iree_status_ignore(status);
      cancelled = true;
      break;
    }
    IREE_RETURN_IF_ERROR(status);
    ++index;
    // Skip whitespace/comments before comma or closing bracket.
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
    // If there's a comma then continue to next element (trailing commas
    // allowed).
    if (!iree_string_view_consume_prefix_char(&str, ',')) break;
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
  }
  // Verify closing bracket (unless cancelled early).
  if (!cancelled && !iree_string_view_starts_with_char(str, ']')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array ]");
  }
  return iree_ok_status();
}

iree_status_t iree_json_enumerate_array_typed(
    iree_string_view_t array_value, iree_json_array_visitor_typed_fn_t visitor,
    void* user_data) {
  iree_string_view_t str = array_value;
  if (!iree_string_view_consume_prefix_char(&str, '[')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array [");
  }
  IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
  iree_host_size_t index = 0;
  bool cancelled = false;
  while (!iree_string_view_is_empty(str)) {
    // Check for end of array.
    if (iree_string_view_starts_with_char(str, ']')) break;
    // Infer the value type from the first character BEFORE consuming.
    // This allows distinguishing string "[" from array [ at the lexical level.
    iree_json_value_type_t type = iree_json_infer_value_type(str.data[0]);
    // Get the array element.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value(&str, &value));
    // Emit the element with its type.
    iree_status_t status = visitor(user_data, index, type, value);
    if (iree_status_is_cancelled(status)) {
      iree_status_ignore(status);
      cancelled = true;
      break;
    }
    IREE_RETURN_IF_ERROR(status);
    ++index;
    // Skip whitespace/comments before comma or closing bracket.
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
    // If there's a comma then continue to next element (trailing commas
    // allowed).
    if (!iree_string_view_consume_prefix_char(&str, ',')) break;
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
  }
  // Verify closing bracket (unless cancelled early).
  if (!cancelled && !iree_string_view_starts_with_char(str, ']')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array ]");
  }
  return iree_ok_status();
}

typedef struct iree_json_array_length_state_t {
  iree_host_size_t count;
} iree_json_array_length_state_t;

static iree_status_t iree_json_array_length_visitor(void* user_data,
                                                    iree_host_size_t index,
                                                    iree_string_view_t value) {
  iree_json_array_length_state_t* state =
      (iree_json_array_length_state_t*)user_data;
  state->count = index + 1;
  (void)value;
  return iree_ok_status();
}

iree_status_t iree_json_array_length(iree_string_view_t array_value,
                                     iree_host_size_t* out_length) {
  *out_length = 0;
  iree_json_array_length_state_t state = {.count = 0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array(
      array_value, iree_json_array_length_visitor, &state));
  *out_length = state.count;
  return iree_ok_status();
}

typedef struct iree_json_array_get_state_t {
  iree_host_size_t target_index;
  iree_string_view_t* out_value;
  bool found;
} iree_json_array_get_state_t;

static iree_status_t iree_json_array_get_visitor(void* user_data,
                                                 iree_host_size_t index,
                                                 iree_string_view_t value) {
  iree_json_array_get_state_t* state = (iree_json_array_get_state_t*)user_data;
  if (index == state->target_index) {
    *state->out_value = value;
    state->found = true;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return iree_ok_status();
}

iree_status_t iree_json_array_get(iree_string_view_t array_value,
                                  iree_host_size_t index,
                                  iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  iree_json_array_get_state_t state = {
      .target_index = index,
      .out_value = out_value,
      .found = false,
  };
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array(
      array_value, iree_json_array_get_visitor, &state));
  if (!state.found) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "array index %" PRIhsz " out of range", index);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// JSONL (JSON Lines) Operations
//===----------------------------------------------------------------------===//

// Skips whitespace and comments, tracking newlines for line number reporting.
// Updates |line_number| to reflect any newlines encountered.
// Handles LF, CRLF, and CR-only line endings.
static iree_status_t iree_json_skip_whitespace_and_comments_tracking_lines(
    iree_string_view_t* str, iree_json_line_number_t* line_number) {
  while (str->size > 0) {
    // Count and skip whitespace characters, tracking newlines.
    while (str->size > 0) {
      char c = str->data[0];
      if (c == ' ' || c == '\t') {
        *str = iree_string_view_remove_prefix(*str, 1);
      } else if (c == '\n') {
        *str = iree_string_view_remove_prefix(*str, 1);
        ++(*line_number);
      } else if (c == '\r') {
        // Handle CRLF as single newline, or CR-only as newline.
        *str = iree_string_view_remove_prefix(*str, 1);
        if (str->size > 0 && str->data[0] == '\n') {
          *str = iree_string_view_remove_prefix(*str, 1);
        }
        ++(*line_number);
      } else {
        break;
      }
    }
    if (str->size == 0) return iree_ok_status();

    // Check for single-line comment: //
    if (str->size >= 2 && str->data[0] == '/' && str->data[1] == '/') {
      // Skip to end of line (but don't skip the newline itself - let the
      // whitespace loop handle it to track line numbers).
      iree_host_size_t i = 2;
      while (i < str->size && str->data[i] != '\n') ++i;
      *str = iree_string_view_remove_prefix(*str, i);
      continue;
    }

    // Check for multi-line comment: /* */
    if (str->size >= 2 && str->data[0] == '/' && str->data[1] == '*') {
      iree_host_size_t i = 2;
      bool found_end = false;
      while (i + 1 < str->size) {
        if (str->data[i] == '\n') {
          ++(*line_number);
        }
        if (str->data[i] == '*' && str->data[i + 1] == '/') {
          *str = iree_string_view_remove_prefix(*str, i + 2);
          found_end = true;
          break;
        }
        ++i;
      }
      if (!found_end) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unterminated /* */ comment");
      }
      continue;
    }

    // No comment found, done skipping.
    break;
  }
  return iree_ok_status();
}

iree_status_t iree_json_enumerate_lines(iree_string_view_t input,
                                        iree_json_line_visitor_fn_t visitor,
                                        void* user_data) {
  // Skip UTF-8 BOM if present at start of file.
  iree_json_skip_bom(&input);
  iree_json_line_number_t line_number = 1;  // 1-based line numbers.
  iree_host_size_t index = 0;               // 0-based entry index.

  while (!iree_string_view_is_empty(input)) {
    // Skip whitespace and comments (may span multiple lines).
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments_tracking_lines(
        &input, &line_number));
    if (iree_string_view_is_empty(input)) break;

    // Record the line number where the value starts.
    iree_json_line_number_t value_line = line_number;

    // Parse the JSON value.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value(&input, &value),
                         "line %" PRIhsz, value_line);

    // Skip any trailing whitespace/comments. In JSONL, each value should be on
    // its own line, so we require that we either hit EOF or advance to a new
    // line after each value.
    iree_json_line_number_t post_value_line = line_number;
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments_tracking_lines(
        &input, &line_number));
    if (!iree_string_view_is_empty(input) && line_number == post_value_line) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "line %" PRIhsz ": multiple values on same line",
                              value_line);
    }

    // Emit the value.
    iree_status_t status = visitor(user_data, value_line, index, value);
    if (iree_status_is_cancelled(status)) {
      iree_status_ignore(status);
      break;
    }
    IREE_RETURN_IF_ERROR(status);
    ++index;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// String Unescaping
//===----------------------------------------------------------------------===//

// Parses a 4-digit hex escape sequence and returns the value.
// |hex| must point to exactly 4 hex characters.
static iree_status_t iree_json_parse_hex4(const char* hex, uint32_t* out) {
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    char c = hex[i];
    value <<= 4;
    if (c >= '0' && c <= '9') {
      value |= (uint32_t)(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      value |= (uint32_t)(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      value |= (uint32_t)(c - 'A' + 10);
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid hex digit '%c' in unicode escape", c);
    }
  }
  *out = value;
  return iree_ok_status();
}

iree_status_t iree_json_unescape_string(iree_string_view_t escaped_string,
                                        iree_host_size_t out_string_capacity,
                                        char* out_string,
                                        iree_host_size_t* out_string_length) {
  iree_host_size_t output_len = 0;
  iree_host_size_t i = 0;

  while (i < escaped_string.size) {
    char c = escaped_string.data[i];

    if (c != '\\') {
      // Regular character - copy directly.
      if (out_string && output_len < out_string_capacity) {
        out_string[output_len] = c;
      }
      ++output_len;
      ++i;
      continue;
    }

    // Escape sequence.
    if (i + 1 >= escaped_string.size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "escape sequence at end of string");
    }
    ++i;  // Skip backslash.
    c = escaped_string.data[i];
    ++i;

    char unescaped;
    switch (c) {
      case '"':
        unescaped = '"';
        break;
      case '\\':
        unescaped = '\\';
        break;
      case '/':
        unescaped = '/';
        break;
      case 'b':
        unescaped = '\b';
        break;
      case 'f':
        unescaped = '\f';
        break;
      case 'n':
        unescaped = '\n';
        break;
      case 'r':
        unescaped = '\r';
        break;
      case 't':
        unescaped = '\t';
        break;
      case 'u': {
        // Unicode escape: \uNNNN or surrogate pair \uD800-\uDBFF \uDC00-\uDFFF.
        if (i + 4 > escaped_string.size) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "truncated unicode escape");
        }
        uint32_t codepoint;
        IREE_RETURN_IF_ERROR(
            iree_json_parse_hex4(&escaped_string.data[i], &codepoint));
        i += 4;

        // Check for surrogate pair (high surrogate).
        if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
          // Must be followed by low surrogate.
          if (i + 6 > escaped_string.size || escaped_string.data[i] != '\\' ||
              escaped_string.data[i + 1] != 'u') {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "high surrogate not followed by low surrogate");
          }
          i += 2;  // Skip \u.
          uint32_t low;
          IREE_RETURN_IF_ERROR(
              iree_json_parse_hex4(&escaped_string.data[i], &low));
          i += 4;
          if (low < 0xDC00 || low > 0xDFFF) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "invalid low surrogate 0x%04X",
                                    (unsigned)low);
          }
          // Combine surrogate pair into codepoint.
          codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
        } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
          // Lone low surrogate is invalid.
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unexpected low surrogate 0x%04X",
                                  (unsigned)codepoint);
        }

        // Encode codepoint as UTF-8.
        char utf8_buf[4];
        int utf8_len = iree_unicode_utf8_encode(codepoint, utf8_buf);
        if (utf8_len == 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid unicode codepoint 0x%X",
                                  (unsigned)codepoint);
        }
        for (int j = 0; j < utf8_len; ++j) {
          if (out_string && output_len < out_string_capacity) {
            out_string[output_len] = utf8_buf[j];
          }
          ++output_len;
        }
        continue;  // Already handled output.
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unrecognized escape sequence '\\%c'", c);
    }

    // Single character escape.
    if (out_string && output_len < out_string_capacity) {
      out_string[output_len] = unescaped;
    }
    ++output_len;
  }

  *out_string_length = output_len;

  // Check if buffer was too small.
  if (out_string && output_len > out_string_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "buffer too small: need %" PRIhsz
                            " bytes, have %" PRIhsz,
                            output_len, out_string_capacity);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Number Parsing Helpers
//===----------------------------------------------------------------------===//

iree_status_t iree_json_parse_int64(iree_string_view_t value, int64_t* out) {
  if (value.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty number");
  }

  // Check for fractional or exponent parts (not valid for integer parsing).
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    char c = value.data[i];
    if (c == '.' || c == 'e' || c == 'E') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected integer, got float");
    }
  }

  // Use iree_string_view_atoi_int64 if available, otherwise manual parsing.
  bool negative = false;
  iree_host_size_t i = 0;
  if (value.data[0] == '-') {
    negative = true;
    ++i;
  }

  if (i >= value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid number");
  }

  uint64_t magnitude = 0;
  while (i < value.size) {
    char c = value.data[i];
    if (!iree_json_is_digit(c)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid digit '%c' in number", c);
    }
    uint64_t digit = (uint64_t)(c - '0');
    // Check for overflow.
    if (magnitude > (UINT64_MAX - digit) / 10) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "integer overflow");
    }
    magnitude = magnitude * 10 + digit;
    ++i;
  }

  if (negative) {
    // INT64_MIN has magnitude 9223372036854775808.
    if (magnitude > (uint64_t)INT64_MAX + 1) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "integer underflow");
    }
    // Special case: INT64_MIN cannot be represented as -(positive int64_t)
    // because its magnitude exceeds INT64_MAX.
    if (magnitude == (uint64_t)INT64_MAX + 1) {
      *out = INT64_MIN;
    } else {
      *out = -(int64_t)magnitude;
    }
  } else {
    if (magnitude > (uint64_t)INT64_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "integer overflow");
    }
    *out = (int64_t)magnitude;
  }

  return iree_ok_status();
}

iree_status_t iree_json_parse_uint64(iree_string_view_t value, uint64_t* out) {
  if (value.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty number");
  }

  // Check for negative, fractional, or exponent parts.
  if (value.data[0] == '-') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected unsigned integer, got negative");
  }
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    char c = value.data[i];
    if (c == '.' || c == 'e' || c == 'E') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected integer, got float");
    }
  }

  uint64_t result = 0;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    char c = value.data[i];
    if (!iree_json_is_digit(c)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid digit '%c' in number", c);
    }
    uint64_t digit = (uint64_t)(c - '0');
    // Check for overflow.
    if (result > (UINT64_MAX - digit) / 10) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "integer overflow");
    }
    result = result * 10 + digit;
  }

  *out = result;
  return iree_ok_status();
}

// Parses a JSON number as a double, locale-independent.
// JSON numbers are always in "C" format: optional '-', digits, optional
// '.digits', optional 'e'/'E' with optional sign and digits. This avoids
// strtod() which is locale-aware and would misparse in non-C locales.
iree_status_t iree_json_parse_double(iree_string_view_t value, double* out) {
  if (value.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty number");
  }

  const char* p = value.data;
  const char* end = value.data + value.size;

  // Handle sign.
  bool negative = false;
  if (p < end && *p == '-') {
    negative = true;
    p++;
  }

  // Parse integer part.
  if (p >= end || !iree_json_is_digit(*p)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid number");
  }
  uint64_t mantissa = 0;
  int mantissa_digits = 0;
  while (p < end && iree_json_is_digit(*p)) {
    if (mantissa_digits <
        18) {  // Avoid overflow, ~18 decimal digits fit in uint64.
      mantissa = mantissa * 10 + (*p - '0');
      mantissa_digits++;
    }
    p++;
  }

  // Parse fractional part.
  int decimal_exponent = 0;
  if (p < end && *p == '.') {
    p++;
    if (p >= end || !iree_json_is_digit(*p)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid fractional part");
    }
    while (p < end && iree_json_is_digit(*p)) {
      if (mantissa_digits < 18) {
        mantissa = mantissa * 10 + (*p - '0');
        mantissa_digits++;
        decimal_exponent--;
      }
      p++;
    }
  }

  // Parse exponent.
  int exponent = 0;
  if (p < end && (*p == 'e' || *p == 'E')) {
    p++;
    bool exp_negative = false;
    if (p < end && (*p == '+' || *p == '-')) {
      exp_negative = (*p == '-');
      p++;
    }
    if (p >= end || !iree_json_is_digit(*p)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid exponent");
    }
    while (p < end && iree_json_is_digit(*p)) {
      exponent = exponent * 10 + (*p - '0');
      if (exponent > 400) {  // Prevent overflow, IEEE 754 max exponent is ~308.
        exponent = 400;
      }
      p++;
    }
    if (exp_negative) exponent = -exponent;
  }

  // Check we consumed all input.
  if (p != end) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid number");
  }

  // Combine mantissa with total exponent.
  int total_exponent = decimal_exponent + exponent;
  double result = (double)mantissa;

  // Apply exponent using powers of 10.
  // Use a lookup table for common exponents to avoid pow() function call.
  static const double pos_powers[] = {
      1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
      1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
  };
  static const double neg_powers[] = {
      1e0,   1e-1,  1e-2,  1e-3,  1e-4,  1e-5,  1e-6,  1e-7,
      1e-8,  1e-9,  1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
      1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22,
  };

  if (total_exponent >= 0) {
    if (total_exponent < (int)(sizeof(pos_powers) / sizeof(pos_powers[0]))) {
      result *= pos_powers[total_exponent];
    } else {
      // Large positive exponent - multiply in steps to avoid overflow.
      while (total_exponent >= 22) {
        result *= 1e22;
        total_exponent -= 22;
      }
      result *= pos_powers[total_exponent];
    }
  } else {
    int abs_exp = -total_exponent;
    if (abs_exp < (int)(sizeof(neg_powers) / sizeof(neg_powers[0]))) {
      result *= neg_powers[abs_exp];
    } else {
      // Large negative exponent - multiply in steps.
      while (abs_exp >= 22) {
        result *= 1e-22;
        abs_exp -= 22;
      }
      result *= neg_powers[abs_exp];
    }
  }

  *out = negative ? -result : result;
  return iree_ok_status();
}

iree_status_t iree_json_parse_bool(iree_string_view_t value, bool* out) {
  if (iree_string_view_equal(value, IREE_SV("true"))) {
    *out = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("false"))) {
    *out = false;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "expected 'true' or 'false', got '%.*s'",
                          (int)value.size, value.data);
}

iree_status_t iree_json_try_lookup_bool(iree_string_view_t object_value,
                                        iree_string_view_t key,
                                        bool default_value, bool* out) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_object_value(object_value, key, &value));
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("null"))) {
    *out = default_value;
    return iree_ok_status();
  }
  return iree_json_parse_bool(value, out);
}

iree_status_t iree_json_try_lookup_int64(iree_string_view_t object_value,
                                         iree_string_view_t key,
                                         int64_t default_value, int64_t* out) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_object_value(object_value, key, &value));
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("null"))) {
    *out = default_value;
    return iree_ok_status();
  }
  return iree_json_parse_int64(value, out);
}

iree_status_t iree_json_try_lookup_string(iree_string_view_t object_value,
                                          iree_string_view_t key,
                                          iree_string_view_t default_value,
                                          char* out_buffer,
                                          iree_host_size_t buffer_capacity,
                                          iree_host_size_t* out_length) {
  // Manually parse to get raw value (with quotes for strings).
  iree_string_view_t str = object_value;
  if (!iree_string_view_consume_prefix_char(&str, '{')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object {");
  }
  IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));

  bool found = false;
  iree_string_view_t raw_value = iree_string_view_empty();
  while (!iree_string_view_is_empty(str)) {
    if (iree_string_view_starts_with_char(str, '}')) break;
    // Parse key.
    iree_string_view_t member_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_string(&str, &member_key));
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
    if (!iree_string_view_consume_prefix_char(&str, ':')) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing object member separator");
    }
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
    // Record raw value span before consuming.
    const char* value_start = str.data;
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value(&str, &value));
    if (iree_string_view_equal(member_key, key)) {
      raw_value = iree_make_string_view(value_start, str.data - value_start);
      found = true;
      break;
    }
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
    if (!iree_string_view_consume_prefix_char(&str, ',')) break;
    IREE_RETURN_IF_ERROR(iree_json_skip_whitespace_and_comments(&str));
  }

  // If not found or null, use default.
  if (!found || iree_string_view_equal(raw_value, IREE_SV("null"))) {
    if (default_value.size > buffer_capacity) {
      *out_length = default_value.size;
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "buffer too small for default value");
    }
    memcpy(out_buffer, default_value.data, default_value.size);
    *out_length = default_value.size;
    return iree_ok_status();
  }

  // Value must be a string (starts with quote).
  if (raw_value.size < 2 || raw_value.data[0] != '"') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected string value for key '%.*s'",
                            (int)key.size, key.data);
  }

  // Strip surrounding quotes and unescape.
  iree_string_view_t content =
      iree_string_view_substr(raw_value, 1, raw_value.size - 2);
  return iree_json_unescape_string(content, buffer_capacity, out_buffer,
                                   out_length);
}

//===----------------------------------------------------------------------===//
// Object Key Validation
//===----------------------------------------------------------------------===//

// Returns true if |key| is in the |allowed_keys| array.
static bool iree_json_is_key_allowed(iree_string_view_t key,
                                     const iree_string_view_t* allowed_keys,
                                     iree_host_size_t allowed_key_count) {
  for (iree_host_size_t i = 0; i < allowed_key_count; ++i) {
    if (iree_string_view_equal(key, allowed_keys[i])) {
      return true;
    }
  }
  return false;
}

// Visitor state for collecting unknown keys.
typedef struct iree_json_validate_keys_state_t {
  const iree_string_view_t* allowed_keys;
  iree_host_size_t allowed_key_count;
  // Unknown keys are collected into a fixed-size buffer.
  // If more unknown keys are found than fit, we just count them.
  iree_string_view_t unknown_keys[8];
  iree_host_size_t unknown_key_count;
} iree_json_validate_keys_state_t;

static iree_status_t iree_json_validate_keys_visitor(void* user_data,
                                                     iree_string_view_t key,
                                                     iree_string_view_t value) {
  (void)value;  // Unused - we only care about keys.
  iree_json_validate_keys_state_t* state =
      (iree_json_validate_keys_state_t*)user_data;
  if (!iree_json_is_key_allowed(key, state->allowed_keys,
                                state->allowed_key_count)) {
    // Record unknown key if we have room.
    if (state->unknown_key_count < IREE_ARRAYSIZE(state->unknown_keys)) {
      state->unknown_keys[state->unknown_key_count] = key;
    }
    ++state->unknown_key_count;
  }
  return iree_ok_status();
}

iree_status_t iree_json_validate_object_keys(
    iree_string_view_t object_value, const iree_string_view_t* allowed_keys,
    iree_host_size_t allowed_key_count) {
  iree_json_validate_keys_state_t state = {
      .allowed_keys = allowed_keys,
      .allowed_key_count = allowed_key_count,
      .unknown_key_count = 0,
  };

  IREE_RETURN_IF_ERROR(iree_json_enumerate_object(
      object_value, iree_json_validate_keys_visitor, &state));

  if (state.unknown_key_count == 0) {
    return iree_ok_status();
  }

  // Build error message listing unknown keys and supported keys.
  // Use a fixed-size buffer for the message.
  char message[512];
  iree_host_size_t message_length = 0;

  // Add unknown keys to message.
  message_length +=
      snprintf(message + message_length, sizeof(message) - message_length,
               "unknown key%s: ", state.unknown_key_count == 1 ? "" : "s");
  iree_host_size_t displayed_count =
      state.unknown_key_count < IREE_ARRAYSIZE(state.unknown_keys)
          ? state.unknown_key_count
          : IREE_ARRAYSIZE(state.unknown_keys);
  for (iree_host_size_t i = 0; i < displayed_count; ++i) {
    if (i > 0) {
      message_length += snprintf(message + message_length,
                                 sizeof(message) - message_length, ", ");
    }
    message_length += snprintf(
        message + message_length, sizeof(message) - message_length, "'%.*s'",
        (int)state.unknown_keys[i].size, state.unknown_keys[i].data);
  }
  if (state.unknown_key_count > displayed_count) {
    message_length += snprintf(
        message + message_length, sizeof(message) - message_length,
        ", ... (+%zu more)", state.unknown_key_count - displayed_count);
  }

  // Add supported keys to message.
  message_length += snprintf(message + message_length,
                             sizeof(message) - message_length, " (supported: ");
  for (iree_host_size_t i = 0; i < allowed_key_count && i < 16; ++i) {
    if (i > 0) {
      message_length += snprintf(message + message_length,
                                 sizeof(message) - message_length, ", ");
    }
    message_length +=
        snprintf(message + message_length, sizeof(message) - message_length,
                 "%.*s", (int)allowed_keys[i].size, allowed_keys[i].data);
  }
  if (allowed_key_count > 16) {
    message_length += snprintf(message + message_length,
                               sizeof(message) - message_length, ", ...");
  }
  message_length +=
      snprintf(message + message_length, sizeof(message) - message_length, ")");

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%.*s",
                          (int)message_length, message);
}

//===----------------------------------------------------------------------===//
// Unicode Codepoint Parsing
//===----------------------------------------------------------------------===//

iree_status_t iree_json_parse_codepoint(iree_string_view_t value,
                                        uint32_t default_codepoint,
                                        uint32_t* out_codepoint) {
  if (value.size == 0) {
    *out_codepoint = default_codepoint;
    return iree_ok_status();
  }

  // First unescape the string to handle \u sequences.
  iree_host_size_t unescaped_length = 0;
  iree_status_t status =
      iree_json_unescape_string(value, 0, NULL, &unescaped_length);
  if (!iree_status_is_ok(status) &&
      !iree_status_is_resource_exhausted(status)) {
    return status;
  }

  // Allocate a small stack buffer for the unescaped character.
  char buffer[8];
  if (unescaped_length > sizeof(buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "codepoint string too long");
  }

  IREE_RETURN_IF_ERROR(iree_json_unescape_string(value, sizeof(buffer), buffer,
                                                 &unescaped_length));

  // Decode the first UTF-8 codepoint.
  if (unescaped_length == 0) {
    *out_codepoint = default_codepoint;
    return iree_ok_status();
  }

  const uint8_t* bytes = (const uint8_t*)buffer;
  if (bytes[0] < 0x80) {
    *out_codepoint = bytes[0];
  } else if ((bytes[0] & 0xE0) == 0xC0 && unescaped_length >= 2) {
    *out_codepoint = ((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F);
  } else if ((bytes[0] & 0xF0) == 0xE0 && unescaped_length >= 3) {
    *out_codepoint = ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) |
                     (bytes[2] & 0x3F);
  } else if ((bytes[0] & 0xF8) == 0xF0 && unescaped_length >= 4) {
    *out_codepoint = ((bytes[0] & 0x07) << 18) | ((bytes[1] & 0x3F) << 12) |
                     ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid UTF-8 in codepoint");
  }

  return iree_ok_status();
}
