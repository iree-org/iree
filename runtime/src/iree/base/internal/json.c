// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/json.h"

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Value Consumption
//===----------------------------------------------------------------------===//

// Forward declaration for mutual recursion.
static iree_status_t iree_json_consume_value_impl(
    iree_string_view_t* str, iree_string_view_t* out_value);

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
    if (i >= str->size || !isdigit((unsigned char)str->data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid number");
    }
    while (i < str->size && isdigit((unsigned char)str->data[i])) {
      ++i;
    }
  }

  // Optional fractional part.
  if (i < str->size && str->data[i] == '.') {
    ++i;
    if (i >= str->size || !isdigit((unsigned char)str->data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid fractional part in number");
    }
    while (i < str->size && isdigit((unsigned char)str->data[i])) {
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
    if (i >= str->size || !isdigit((unsigned char)str->data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid exponent in number");
    }
    while (i < str->size && isdigit((unsigned char)str->data[i])) {
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

iree_status_t iree_json_consume_object(iree_string_view_t* str,
                                       iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  const char* start = str->data;
  if (!iree_string_view_consume_prefix(str, IREE_SV("{"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object {");
  }
  *str = iree_string_view_trim(*str);
  while (!iree_string_view_is_empty(*str)) {
    // Check for end of object.
    if (iree_string_view_starts_with(*str, IREE_SV("}"))) break;
    // Try to parse key string.
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_string(str, &key));
    *str = iree_string_view_trim(*str);
    // Expect : separator.
    if (!iree_string_view_consume_prefix(str, IREE_SV(":"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing object member separator");
    }
    // Scan ahead to get the value span.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value_impl(str, &value));
    // Trim after value to handle whitespace before comma or closing brace.
    *str = iree_string_view_trim(*str);
    // If there's a comma then we expect another member.
    if (!iree_string_view_consume_prefix(str, IREE_SV(","))) {
      // No comma - must be end of object (trailing commas are invalid JSON).
      break;
    }
    *str = iree_string_view_trim(*str);
    // After comma, must have another member (not closing brace).
    if (iree_string_view_starts_with(*str, IREE_SV("}"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "trailing comma in object");
    }
  }
  // Save end pointer BEFORE consuming `}` (consume_prefix returns empty view
  // with NULL data when string becomes empty).
  const char* end = str->data + 1;
  if (!iree_string_view_consume_prefix(str, IREE_SV("}"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object }");
  }
  *out_value = iree_make_string_view(start, end - start);
  return iree_ok_status();
}

iree_status_t iree_json_consume_array(iree_string_view_t* str,
                                      iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  const char* start = str->data;
  if (!iree_string_view_consume_prefix(str, IREE_SV("["))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array [");
  }
  *str = iree_string_view_trim(*str);
  while (!iree_string_view_is_empty(*str)) {
    // Check for end of array.
    if (iree_string_view_starts_with(*str, IREE_SV("]"))) break;
    // Get the array element.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value_impl(str, &value));
    // Trim after value to handle whitespace before comma or closing bracket.
    *str = iree_string_view_trim(*str);
    // If there's a comma then we expect another element.
    if (!iree_string_view_consume_prefix(str, IREE_SV(","))) {
      // No comma - must be end of array (trailing commas are invalid JSON).
      break;
    }
    *str = iree_string_view_trim(*str);
    // After comma, must have another element (not closing bracket).
    if (iree_string_view_starts_with(*str, IREE_SV("]"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "trailing comma in array");
    }
  }
  // Save end pointer BEFORE consuming `]` (consume_prefix returns empty view
  // with NULL data when string becomes empty).
  const char* end = str->data + 1;
  if (!iree_string_view_consume_prefix(str, IREE_SV("]"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array ]");
  }
  *out_value = iree_make_string_view(start, end - start);
  return iree_ok_status();
}

static iree_status_t iree_json_consume_value_impl(
    iree_string_view_t* str, iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  *str = iree_string_view_trim(*str);
  if (str->size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected JSON value");
  }
  switch (str->data[0]) {
    case '"':
      return iree_json_consume_string(str, out_value);
    case '{':
      return iree_json_consume_object(str, out_value);
    case '[':
      return iree_json_consume_array(str, out_value);
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
  return iree_json_consume_value_impl(str, out_value);
}

//===----------------------------------------------------------------------===//
// Object Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_json_enumerate_object(iree_string_view_t object_value,
                                         iree_json_object_visitor_fn_t visitor,
                                         void* user_data) {
  iree_string_view_t str = object_value;
  if (!iree_string_view_consume_prefix(&str, IREE_SV("{"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object {");
  }
  str = iree_string_view_trim(str);
  bool cancelled = false;
  while (!iree_string_view_is_empty(str)) {
    // Check for end of object.
    if (iree_string_view_starts_with(str, IREE_SV("}"))) break;
    // Try to parse key string.
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_string(&str, &key));
    str = iree_string_view_trim(str);
    // Expect : separator.
    if (!iree_string_view_consume_prefix(&str, IREE_SV(":"))) {
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
    // Trim after value to handle whitespace before comma or closing brace.
    str = iree_string_view_trim(str);
    // If there's a comma then we expect another member.
    if (!iree_string_view_consume_prefix(&str, IREE_SV(","))) {
      // No comma - must be end of object.
      break;
    }
    str = iree_string_view_trim(str);
    // After comma, must have another member (trailing commas are invalid).
    if (iree_string_view_starts_with(str, IREE_SV("}"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "trailing comma in object");
    }
  }
  // Verify closing brace (unless cancelled early).
  if (!cancelled && !iree_string_view_starts_with(str, IREE_SV("}"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object }");
  }
  return iree_ok_status();
}

typedef struct iree_json_lookup_object_value_state_t {
  iree_string_view_t key;
  iree_string_view_t* value;
} iree_json_lookup_object_value_state_t;

static iree_status_t iree_json_lookup_object_value_visitor(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  iree_json_lookup_object_value_state_t* state =
      (iree_json_lookup_object_value_state_t*)user_data;
  if (iree_string_view_equal(key, state->key)) {
    *state->value = value;
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
  if (!iree_string_view_consume_prefix(&str, IREE_SV("["))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array [");
  }
  str = iree_string_view_trim(str);
  iree_host_size_t index = 0;
  bool cancelled = false;
  while (!iree_string_view_is_empty(str)) {
    // Check for end of array.
    if (iree_string_view_starts_with(str, IREE_SV("]"))) break;
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
    // Trim after value to handle whitespace before comma or closing bracket.
    str = iree_string_view_trim(str);
    // If there's a comma then we expect another element.
    if (!iree_string_view_consume_prefix(&str, IREE_SV(","))) {
      // No comma - must be end of array.
      break;
    }
    str = iree_string_view_trim(str);
    // After comma, must have another element (trailing commas are invalid).
    if (iree_string_view_starts_with(str, IREE_SV("]"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "trailing comma in array");
    }
  }
  // Verify closing bracket (unless cancelled early).
  if (!cancelled && !iree_string_view_starts_with(str, IREE_SV("]"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array ]");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// JSONL (JSON Lines) Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_json_enumerate_lines(iree_string_view_t input,
                                        iree_json_line_visitor_fn_t visitor,
                                        void* user_data) {
  iree_json_line_number_t line_number = 0;  // Will be 1-based after increment.
  iree_host_size_t index = 0;               // 0-based entry index.
  while (!iree_string_view_is_empty(input)) {
    ++line_number;

    // Find the end of the current line.
    iree_string_view_t line;
    iree_host_size_t newline_pos = iree_string_view_find_char(input, '\n', 0);
    if (newline_pos == IREE_STRING_VIEW_NPOS) {
      // Last line (no trailing newline).
      line = input;
      input = iree_string_view_empty();
    } else {
      line = iree_string_view_substr(input, 0, newline_pos);
      input =
          iree_string_view_substr(input, newline_pos + 1, IREE_HOST_SIZE_MAX);
    }

    // Strip trailing CR for CRLF line endings (Windows).
    if (line.size > 0 && line.data[line.size - 1] == '\r') {
      line = iree_string_view_substr(line, 0, line.size - 1);
    }

    // Skip empty lines and whitespace-only lines.
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }

    // Parse the JSON value on this line.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value(&line, &value),
                         "line %" PRIhsz, line_number);

    // Verify no trailing content after the value (except whitespace).
    line = iree_string_view_trim(line);
    if (!iree_string_view_is_empty(line)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "line %" PRIhsz ": trailing content after value",
                              line_number);
    }

    // Emit the value.
    iree_status_t status = visitor(user_data, line_number, index, value);
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
    if (!isdigit((unsigned char)c)) {
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
    if (!isdigit((unsigned char)c)) {
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

iree_status_t iree_json_parse_double(iree_string_view_t value, double* out) {
  if (value.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty number");
  }

  // Need a null-terminated string for strtod.
  // Use a stack buffer for small numbers, which covers all realistic cases.
  char stack_buf[64];
  char* buf = stack_buf;
  if (value.size >= sizeof(stack_buf)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "number too long for parsing");
  }
  memcpy(buf, value.data, value.size);
  buf[value.size] = '\0';

  char* endptr;
  errno = 0;
  double result = strtod(buf, &endptr);

  if (errno == ERANGE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "float out of range");
  }
  if (endptr != buf + value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid number");
  }

  *out = result;
  return iree_ok_status();
}
