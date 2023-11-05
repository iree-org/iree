// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/safetensors/safetensors_format.h"

#include <ctype.h>

// File format:
// - uint64_t header_length;
// - uint8_t header_json[header_length];
// - uint8_t remaining_data[];
//
// JSON:
//   {
//     "TENSOR_NAME": {
//       "dtype": "F16",
//       "shape": [1, 16, 256],
//       "data_offsets": [BEGIN, END]
//     },
//     "NEXT_TENSOR_NAME": {...}
//   }
//
// The BEGIN offset is relative to the end of the header, not the file.
// The END offset is oddly BEGIN+length such that length=END-BEGIN.
//
// Note that for such a fixed file format JSON is overkill and we don't want to
// pull in a JSON parser just to get the data offsets. We parse the strings the
// old fashioned way while wishing they were just numbers and bail if anything
// looks suspect.
//
// Here's a real JSON blob from a test file (formatted to add whitespace):
// <<8 byte header with a count of all bytes including trailing whitespace>>
// {
//   "attention": {
//     "dtype": "F32",
//     "shape": [
//       2,
//       3
//     ],
//     "data_offsets": [
//       0,
//       24
//     ]
//   },
//   "embedding": {
//     "dtype": "F32",
//     "shape": [
//       2,
//       2
//     ],
//     "data_offsets": [
//       24,
//       40
//     ]
//   }
// }
// <<trailing whitespace>>
// <<40 bytes of data>>
//
// Basic JSON spec (per json.org):
//  value:
//   object
//   array
//   string
//   number
//   "true"
//   "false"
//   "null"
//  object:
//   '{' ws '}'
//   '{' members '}'
//  members:
//   member
//   member ',' members
//  member:
//   ws string ws ':' element
//  array:
//   '[' ws ']'
//   '[' elements ']'
//  elements:
//   element
//   element ',' elements
//  element:
//   ws value ws
//  string:
//   '"' characters '"'
//  characters:
//   ""
//   character characters
//  character:
//   0x0020 . 0x10FFFF - '"' - '\'
//   '\' escape
//  escape:
//   '"' '\' '/' 'b' 'f' 'n' 'r' 't'
//   'u' hex hex hex hex
//  hex:
//   digit
//   'A' . 'F'
//   'a' . 'f'
//  number:
//   integer fraction exponent
//  ws:
//   ""
//   0x0020 ws
//   0x000A ws
//   0x000D ws
//   0x0009 ws

static iree_status_t iree_json_consume_value(iree_string_view_t* str,
                                             iree_string_view_t* out_value);

// Consumes |keyword| from |str| and returns it, updating |str| to point
// immediately after it.
static iree_status_t iree_json_consume_keyword(iree_string_view_t* str,
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

// Consumes a number from |str| and returns as declared, updating |str| to
// point immediately after the last character that could compose the number.
// Assumes the input starts with `number` in the spec.
static iree_status_t iree_json_consume_number(iree_string_view_t* str,
                                              iree_string_view_t* out_value) {
  // TODO(benvanik): support real numbers - for now we only handle integers
  // because we are lazy. We scan for digits 0-9 until any non-digit is hit and
  // then call it good.
  iree_host_size_t break_pos = 0;
  for (iree_host_size_t i = 0; i < str->size; ++i) {
    if (!isdigit(str->data[i])) {
      break_pos = i;
      break;
    }
  }
  if (break_pos == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid number");
  }
  *out_value = iree_string_view_substr(*str, 0, break_pos);
  *str = iree_string_view_remove_prefix(*str, break_pos);
  return iree_ok_status();
}

// Consumes a string from |str| and returns it unquoted, updating |str| to
// point immediately after the trailing double quote of the string.
// Assumes the input starts with `string` in the spec.
static iree_status_t iree_json_consume_string(iree_string_view_t* str,
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
          i += 4;
          break;
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unrecognized string escape code %c", c);
      }
    }
  }
  *out_value = iree_string_view_substr(*str, start, end - start);
  *str = iree_string_view_substr(*str, end + 1, IREE_HOST_SIZE_MAX);
  return iree_ok_status();
}

// Consumes an object and all its descendents from |str| and returns it with
// braces, updating |str| to point immediately after the trailing `}`.
// Assumes the input starts with `object` in the spec.
static iree_status_t iree_json_consume_object(iree_string_view_t* str,
                                              iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  const char* start = str->data;
  if (!iree_string_view_consume_prefix(str, IREE_SV("{"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object {");
  }
  *str = iree_string_view_trim(*str);
  while (!iree_string_view_is_empty(*str)) {
    // Check for end of object.
    if (iree_string_view_consume_prefix(str, IREE_SV("}"))) break;
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
    IREE_RETURN_IF_ERROR(iree_json_consume_value(str, &value));
    // If there's a comma then we expect another value.
    if (!iree_string_view_consume_prefix(str, IREE_SV(","))) break;
    *str = iree_string_view_trim(*str);
  }
  if (!iree_string_view_consume_prefix(str, IREE_SV("}"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object }");
  }
  const char* end = str->data;
  *out_value = iree_make_string_view(start, end - start);
  return iree_ok_status();
}

// Consumes an array and all its descendents from |str| and returns it with
// brackets, updating |str| to point immediately after the trailing `]`.
// Assumes the input starts with `array` in the spec.
static iree_status_t iree_json_consume_array(iree_string_view_t* str,
                                             iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  const char* start = str->data;
  if (!iree_string_view_consume_prefix(str, IREE_SV("["))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array [");
  }
  *str = iree_string_view_trim(*str);
  while (!iree_string_view_is_empty(*str)) {
    // Check for end of array.
    if (iree_string_view_consume_prefix(str, IREE_SV("]"))) break;
    // Get the array element.
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_value(str, &value));
    // If there's a comma then we expect another value.
    if (!iree_string_view_consume_prefix(str, IREE_SV(","))) break;
    *str = iree_string_view_trim(*str);
  }
  if (!iree_string_view_consume_prefix(str, IREE_SV("]"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing array ]");
  }
  const char* end = str->data;
  *out_value = iree_make_string_view(start, end - start);
  return iree_ok_status();
}

// Consumes a value from |str| and returns it, updating |str| to point
// immediately after it.
// Assumes the input starts with `value` in the spec.
static iree_status_t iree_json_consume_value(iree_string_view_t* str,
                                             iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  *str = iree_string_view_trim(*str);
  if (str->size == 0) return iree_ok_status();
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

typedef iree_status_t(IREE_API_PTR* iree_json_object_enumerator_fn_t)(
    void* user_data, iree_string_view_t key, iree_string_view_t value);

// Enumerates all key-value pairs in the given object |str|.
// Assumes that the input matches `object` in the spec (`{` and `}` at edges).
// |enumerator| can return IREE_STATUS_CANCELLED to skip all further entries.
static iree_status_t iree_json_enumerate_object(
    iree_string_view_t str, iree_json_object_enumerator_fn_t enumerator,
    void* user_data) {
  if (!iree_string_view_consume_prefix(&str, IREE_SV("{"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing object {");
  }
  str = iree_string_view_trim(str);
  while (!iree_string_view_is_empty(str)) {
    // Check for end of object.
    if (iree_string_view_consume_prefix(&str, IREE_SV("}"))) break;
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
    iree_status_t status = enumerator(user_data, key, value);
    if (iree_status_is_cancelled(status)) {
      iree_status_ignore(status);
      break;
    }
    IREE_RETURN_IF_ERROR(status);
    // If there's a comma then we expect another value.
    if (!iree_string_view_consume_prefix(&str, IREE_SV(","))) break;
    str = iree_string_view_trim(str);
  }
  return iree_ok_status();
}

typedef struct iree_json_lookup_object_value_state_t {
  iree_string_view_t key;
  iree_string_view_t* value;
} iree_json_lookup_object_value_state_t;
static iree_status_t iree_json_lookup_object_value_enumerator(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  iree_json_lookup_object_value_state_t* state =
      (iree_json_lookup_object_value_state_t*)user_data;
  if (iree_string_view_equal(key, state->key)) {
    *state->value = value;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return iree_ok_status();
}

// Finds a directly nested |key| in the JSON |object_str| and returns its value.
// Example:
//   {"foo": {"bar": true}, "taco": 51}
//   iree_json_lookup_object_value("foo") -> `{"bar": true}`
//   iree_json_lookup_object_value("taco") -> `51`
static iree_status_t iree_json_lookup_object_value(
    iree_string_view_t object_str, iree_string_view_t key,
    iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  iree_json_lookup_object_value_state_t state = {
      .key = key,
      .value = out_value,
  };
  return iree_json_enumerate_object(
      object_str, iree_json_lookup_object_value_enumerator, &state);
}

// Parses a `[begin, end]` JSON array.
static bool iree_io_parse_json_data_offsets(iree_string_view_t data_offsets_str,
                                            uint64_t* out_begin,
                                            uint64_t* out_end) {
  if (!iree_string_view_consume_prefix(&data_offsets_str, IREE_SV("[")) ||
      !iree_string_view_consume_suffix(&data_offsets_str, IREE_SV("]"))) {
    return false;
  }
  iree_string_view_t begin_str = iree_string_view_empty();
  iree_string_view_t end_str = iree_string_view_empty();
  if (iree_string_view_split(iree_string_view_trim(data_offsets_str), ',',
                             &begin_str, &end_str) == -1) {
    return false;
  }
  return iree_string_view_atoi_uint64(iree_string_view_trim(begin_str),
                                      out_begin) &&
         iree_string_view_atoi_uint64(iree_string_view_trim(end_str), out_end);
}

typedef struct iree_io_enumerate_safetensors_entry_state_t {
  iree_io_file_handle_t* file_handle;
  uint64_t base_offset;
  uint64_t data_size;
  iree_io_parameter_index_t* index;
} iree_io_enumerate_safetensors_entry_state_t;

// Enumerates the outer safetensors header JSON object and emits entries to the
// |index|. |key| will be the tensor name (what we call a parameter key) and
// |value| will be the entry object we'll need to extract info from.
//
// Each entry in the dictionary looks something like this, note the order of
// the fields is undefined and there may be some we ignore:
//   "TENSOR_NAME": {
//     "dtype": "F16",
//     "shape": [1, 16, 256],
//     "data_offsets": [BEGIN, END]
//   },  <-- optional (omitted at end)
static iree_status_t iree_io_enumerate_safetensors_entries(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  iree_io_enumerate_safetensors_entry_state_t* entry_state =
      (iree_io_enumerate_safetensors_entry_state_t*)user_data;

  // Ignore special "__metadata__" entry. We ignore it for now.
  if (iree_string_view_equal(key, IREE_SV("__metadata__"))) {
    return iree_ok_status();
  }

  // Lookup the data offsets array.
  iree_string_view_t data_offsets_str = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      value, IREE_SV("data_offsets"), &data_offsets_str));

  // Extract the data offsets from the array and verify they are in range.
  uint64_t begin = 0;
  uint64_t end = 0;
  if (!iree_io_parse_json_data_offsets(data_offsets_str, &begin, &end)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to parse entry data offsets `%.*s`",
                            (int)data_offsets_str.size, data_offsets_str.data);
  }
  if (begin > end || end > entry_state->data_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry has data offsets outside of the "
                            "available data (begin=%" PRIu64 ", end=%" PRIu64
                            ", available=%" PRIu64 ")",
                            begin, end, entry_state->data_size);
  }

  // Add entry to the index.
  iree_io_parameter_index_entry_t entry = {
      .key = key,
      .metadata = iree_const_byte_span_empty(),
      .length = end - begin,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE,
      .storage =
          {
              .file =
                  {
                      .handle = entry_state->file_handle,
                      .offset = entry_state->base_offset + begin,
                  },
          },
  };
  return iree_io_parameter_index_add(entry_state->index, &entry);
}

IREE_API_EXPORT iree_status_t iree_io_parse_safetensors_index_from_memory(
    iree_io_file_handle_t* file_handle, iree_const_byte_span_t file_contents,
    iree_io_parameter_index_t* index) {
  // Reads the header JSON blob out of the file contents and calculates the base
  // offset that all data ranges are relative to. Verifies that the header and
  // base offset is in range but each entry data range still needs to be
  // verified.
  uint64_t remaining_bytes = file_contents.data_length;
  uint64_t header_length = 0;
  if (remaining_bytes < sizeof(header_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "insufficient capacity for safetensors header "
                            "length (need at least %" PRIhsz
                            " bytes but have %" PRIu64 ")",
                            sizeof(header_length), remaining_bytes);
  }
  header_length =
      iree_unaligned_load_le_u64((const uint64_t*)file_contents.data);
  remaining_bytes -= sizeof(header_length);
  if (remaining_bytes < header_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "insufficient capacity for safetensors header "
                            "contents (declared as %" PRIu64
                            " but only %" PRIu64 " bytes available)",
                            header_length, remaining_bytes);
  }
  const iree_string_view_t header_json = iree_make_string_view(
      (const char*)file_contents.data + sizeof(header_length),
      (iree_host_size_t)header_length);
  const uint64_t base_offset = sizeof(header_length) + header_length;
  remaining_bytes -= header_length;

  // Parses a safetensors |header_json| blob and emits entries to |index|.
  // Each entry is bounds checked against the |data_size| of the file (bytes
  // excluding the header, relative to |base_offset|).
  iree_io_enumerate_safetensors_entry_state_t enumerate_state = {
      .file_handle = file_handle,
      .base_offset = base_offset,
      .data_size = remaining_bytes,
      .index = index,
  };
  return iree_json_enumerate_object(
      header_json, iree_io_enumerate_safetensors_entries, &enumerate_state);
}

IREE_API_EXPORT iree_status_t iree_io_parse_safetensors_index(
    iree_io_file_handle_t* file_handle, iree_io_parameter_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Today we only support memory files.
  // TODO(benvanik): support iree_io_stream_t wrapping for parsing the index.
  if (iree_io_file_handle_type(file_handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-memory safetensors files not yet supported");
  }
  iree_byte_span_t host_allocation =
      iree_io_file_handle_primitive(file_handle).value.host_allocation;

  iree_status_t status = iree_io_parse_safetensors_index_from_memory(
      file_handle,
      iree_make_const_byte_span(host_allocation.data,
                                host_allocation.data_length),
      index);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
