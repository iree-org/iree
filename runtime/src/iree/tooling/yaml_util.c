// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/yaml_util.h"

iree_status_t iree_status_from_yaml_parser_error(yaml_parser_t* parser) {
  // TODO(benvanik): copy parser.error to status.
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "yaml_parser_load failed");
}

iree_string_view_t iree_yaml_node_as_string(yaml_node_t* node) {
  if (!node || node->type != YAML_SCALAR_NODE) return iree_string_view_empty();
  return iree_make_string_view(node->data.scalar.value,
                               node->data.scalar.length);
}

bool iree_yaml_string_equal(yaml_node_t* node, iree_string_view_t value) {
  return iree_string_view_equal(iree_yaml_node_as_string(node), value);
}

iree_status_t iree_yaml_mapping_try_find(yaml_document_t* document,
                                         yaml_node_t* node,
                                         iree_string_view_t key,
                                         yaml_node_t** out_value) {
  *out_value = NULL;
  if (!node) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "invalid node");
  }
  if (node->type != YAML_MAPPING_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected mapping node",
                            node->start_mark.line);
  }
  for (yaml_node_pair_t* pair = node->data.mapping.pairs.start;
       pair != node->data.mapping.pairs.top; ++pair) {
    yaml_node_t* key_node = yaml_document_get_node(document, pair->key);
    if (iree_yaml_string_equal(key_node, key)) {
      *out_value = yaml_document_get_node(document, pair->value);
      if (!*out_value) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "(%zu): mapping entry has no value",
                                key_node->start_mark.line);
      }
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t iree_yaml_mapping_find(yaml_document_t* document,
                                     yaml_node_t* node, iree_string_view_t key,
                                     yaml_node_t** out_value) {
  IREE_RETURN_IF_ERROR(
      iree_yaml_mapping_try_find(document, node, key, out_value));
  if (!*out_value) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no mapping found for key '%.*s'", (int)key.size,
                            key.data);
  }
  return iree_ok_status();
}

// base64 incremental decoding with support for interior whitespace (as required
// by the general YAML encoding convention).
//
// YAML !binary encoding:
// https://yaml.org/type/binary.html
//
// Source:
// https://en.wikibooks.org/wiki/Algorithm_Implementation/Miscellaneous/Base64#C_2
static const uint8_t iree_yaml_base64_decode_table[] = {
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 64, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 62, 66, 66, 66, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 66, 66, 66, 65, 66, 66, 66, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 66, 66, 66, 66,
    66, 66, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66,
};

#define IREE_YAML_BASE64_WHITESPACE 64
#define IREE_YAML_BASE64_EQUALS 65
#define IREE_YAML_BASE64_INVALID 66

size_t iree_yaml_base64_calculate_size(iree_string_view_t source) {
  uint32_t block = 0;
  size_t block_length = 0;
  size_t decoded_length = 0;
  size_t i = 0;
  while (i < source.size) {
    uint8_t c = iree_yaml_base64_decode_table[(uint8_t)source.data[i++]];
    if (c == IREE_YAML_BASE64_WHITESPACE) {
      // Skip whitespace.
      continue;
    } else if (c == IREE_YAML_BASE64_INVALID) {
      // Invalid base64 character.
      return SIZE_MAX;
    } else if (c == IREE_YAML_BASE64_EQUALS) {
      // End-of-data padding ('=').
      break;
    }
    // Additional byte for the current 4-byte block.
    block = block << 6 | c;
    ++block_length;
    if (block_length == 4) {
      // Flush 4-byte block.
      decoded_length += 3;
      block = 0;
      block_length = 0;
    }
  }
  if (block_length == 3) {
    // 2 bytes leftover.
    decoded_length += 2;
  } else if (block_length == 2) {
    // 1 byte leftover.
    ++decoded_length;
  }
  return decoded_length;
}

iree_status_t iree_yaml_base64_decode(iree_string_view_t source,
                                      iree_byte_span_t target) {
  uint32_t block = 0;
  size_t block_length = 0;
  size_t decoded_length = 0;
  size_t i = 0;
  uint8_t* p = target.data;
  while (i < source.size) {
    uint8_t c = iree_yaml_base64_decode_table[(uint8_t)source.data[i++]];
    if (c == IREE_YAML_BASE64_WHITESPACE) {
      // Skip whitespace.
      continue;
    } else if (c == IREE_YAML_BASE64_INVALID) {
      // Invalid base64 character.
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid base64 character");
    } else if (c == IREE_YAML_BASE64_EQUALS) {
      // End-of-data padding ('=').
      break;
    }
    // Additional byte for the current 4-byte block.
    block = block << 6 | c;
    ++block_length;
    if (block_length == 4) {
      // Flush 4-byte block.
      decoded_length += 3;
      if (decoded_length > target.data_length) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "base64 target buffer overflow");
      }
      *(p++) = (block >> 16) & 0xFF;
      *(p++) = (block >> 8) & 0xFF;
      *(p++) = block & 0xFF;
      block = 0;
      block_length = 0;
    }
  }
  if (block_length == 3) {
    // 2 bytes leftover.
    decoded_length += 2;
    if (decoded_length > target.data_length) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "base64 target buffer overflow");
    }
    *(p++) = (block >> 10) & 0xFF;
    *(p++) = (block >> 2) & 0xFF;
  } else if (block_length == 2) {
    // 1 byte leftover.
    ++decoded_length;
    if (decoded_length > target.data_length) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "base64 target buffer overflow");
    }
    *(p++) = (block >> 4) & 0xFF;
  }
  return iree_ok_status();
}
