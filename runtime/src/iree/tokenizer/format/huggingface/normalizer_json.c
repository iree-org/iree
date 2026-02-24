// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/normalizer_json.h"

#include "iree/base/internal/json.h"
#include "iree/tokenizer/normalizer/bert.h"
#include "iree/tokenizer/normalizer/lowercase.h"
#include "iree/tokenizer/normalizer/nfc.h"
#include "iree/tokenizer/normalizer/nfd.h"
#include "iree/tokenizer/normalizer/nfkd.h"
#include "iree/tokenizer/normalizer/precompiled.h"
#include "iree/tokenizer/normalizer/prepend.h"
#include "iree/tokenizer/normalizer/regex_replace.h"
#include "iree/tokenizer/normalizer/replace.h"
#include "iree/tokenizer/normalizer/sequence.h"
#include "iree/tokenizer/normalizer/strip.h"
#include "iree/tokenizer/normalizer/strip_accents.h"

//===----------------------------------------------------------------------===//
// Base64 Decoding
//===----------------------------------------------------------------------===//

// Base64 decoding table (6 bits per character).
// -1 = invalid, -2 = padding ('=').
static const int8_t iree_tokenizer_base64_decode_table[256] = {
    // clang-format off
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 0x00-0x0F
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 0x10-0x1F
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,  // 0x20-0x2F (+, /)
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,  // 0x30-0x3F (0-9, =)
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,  // 0x40-0x4F (A-O)
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,  // 0x50-0x5F (P-Z)
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,  // 0x60-0x6F (a-o)
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,  // 0x70-0x7F (p-z)
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 0x80-0xFF (invalid)
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    // clang-format on
};

// Decodes base64-encoded data into a caller-allocated buffer.
static iree_status_t iree_tokenizer_base64_decode(
    iree_string_view_t encoded, uint8_t* out_data,
    iree_host_size_t out_capacity, iree_host_size_t* out_length) {
  if (encoded.size == 0) {
    *out_length = 0;
    return iree_ok_status();
  }

  // Validate base64 input length (must be multiple of 4, minimum 4 characters).
  if (encoded.size < 4 || (encoded.size % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid base64 length %" PRIhsz
                            " (must be non-zero multiple of 4)",
                            encoded.size);
  }

  // Calculate required output size.
  iree_host_size_t padding = 0;
  if (encoded.data[encoded.size - 1] == '=') padding++;
  if (encoded.data[encoded.size - 2] == '=') padding++;
  iree_host_size_t base_decoded = (encoded.size / 4) * 3;
  if (padding > base_decoded) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid base64: padding exceeds decoded size");
  }
  iree_host_size_t decoded_size = base_decoded - padding;

  if (out_capacity < decoded_size) {
    *out_length = decoded_size;
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "base64 output buffer too small");
  }

  iree_host_size_t out_position = 0;
  uint32_t accumulator = 0;
  int bits = 0;

  for (iree_host_size_t i = 0; i < encoded.size; ++i) {
    uint8_t c = (uint8_t)encoded.data[i];
    int8_t value = iree_tokenizer_base64_decode_table[c];
    if (value == -1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid base64 character at position %" PRIhsz,
                              i);
    }
    if (value == -2) continue;  // Padding.
    accumulator = (accumulator << 6) | (uint32_t)value;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      if (out_position < out_capacity) {
        out_data[out_position++] = (uint8_t)(accumulator >> bits);
      }
    }
  }

  *out_length = out_position;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Precompiled Normalizer
//===----------------------------------------------------------------------===//

// Parses a Precompiled normalizer (SentencePiece):
// {
//   "type": "Precompiled",
//   "precompiled_charsmap": "base64-encoded-data"
// }
static iree_status_t iree_tokenizer_parse_precompiled_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  *out_normalizer = NULL;
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("precompiled_charsmap"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Look up the precompiled_charsmap field.
  iree_string_view_t charsmap_b64 = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      json, IREE_SV("precompiled_charsmap"), &charsmap_b64));

  // Handle empty charsmap - return NULL (identity/passthrough).
  if (charsmap_b64.size == 0) {
    *out_normalizer = NULL;
    return iree_ok_status();
  }

  // Validate base64 length.
  if (charsmap_b64.size < 4 || (charsmap_b64.size % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid base64 charsmap length %" PRIhsz,
                            charsmap_b64.size);
  }

  // Calculate decoded size.
  iree_host_size_t padding = 0;
  if (charsmap_b64.data[charsmap_b64.size - 1] == '=') padding++;
  if (charsmap_b64.data[charsmap_b64.size - 2] == '=') padding++;
  iree_host_size_t base_decoded = (charsmap_b64.size / 4) * 3;
  if (padding > base_decoded) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid base64 charsmap padding");
  }
  iree_host_size_t decoded_size = base_decoded - padding;

  // Allocate temporary buffer for decoded data.
  uint8_t* decoded_data = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, decoded_size, (void**)&decoded_data);
  if (!iree_status_is_ok(status)) return status;

  // Decode base64.
  iree_host_size_t actual_length = 0;
  status = iree_tokenizer_base64_decode(charsmap_b64, decoded_data,
                                        decoded_size, &actual_length);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, decoded_data);
    return status;
  }

  // Allocate the precompiled normalizer from decoded data.
  status = iree_tokenizer_precompiled_normalizer_allocate(
      iree_make_const_byte_span(decoded_data, actual_length), allocator,
      out_normalizer);

  // Free temporary buffer (normalizer copies what it needs).
  iree_allocator_free(allocator, decoded_data);

  return status;
}

//===----------------------------------------------------------------------===//
// Sequence Normalizer
//===----------------------------------------------------------------------===//

// Context for parsing sequence normalizer children.
typedef struct iree_tokenizer_normalizer_sequence_parse_context_t {
  iree_allocator_t allocator;
  iree_tokenizer_normalizer_t** children;
  iree_host_size_t child_count;
  iree_host_size_t parsed_count;
} iree_tokenizer_normalizer_sequence_parse_context_t;

// Visitor callback for parsing sequence normalizer children.
static iree_status_t iree_tokenizer_parse_normalizer_sequence_child_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t value) {
  iree_tokenizer_normalizer_sequence_parse_context_t* context =
      (iree_tokenizer_normalizer_sequence_parse_context_t*)user_data;
  if (index >= context->child_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "array index %zu exceeds expected count %zu", index,
                            context->child_count);
  }
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      value, context->allocator, &context->children[index]);
  if (iree_status_is_ok(status)) {
    ++context->parsed_count;
  } else {
    status = iree_status_annotate_f(status, "in Sequence normalizer child %zu",
                                    index);
  }
  return status;
}

// Parses a Sequence normalizer.
// JSON structure:
//   {
//     "type": "Sequence",
//     "normalizers": [
//       {"type": "NFC"},
//       {"type": "Lowercase"},
//       ...
//     ]
//   }
//
// NULL children (from no-op normalizers) are filtered out.
static iree_status_t iree_tokenizer_parse_sequence_normalizer(
    iree_string_view_t normalizer_value, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  *out_normalizer = NULL;
  iree_status_t status = iree_ok_status();

  // Look up the normalizers array.
  iree_string_view_t normalizers_value = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = iree_json_lookup_object_value(
        normalizer_value, IREE_SV("normalizers"), &normalizers_value);
  }

  // Get array length.
  iree_host_size_t json_child_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_json_array_length(normalizers_value, &json_child_count);
  }

  // Validate count against maximum depth.
  if (iree_status_is_ok(status) &&
      json_child_count > IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Sequence normalizer has %zu children, maximum is %d", json_child_count,
        IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH);
  }

  // Allocate array for child normalizer pointers.
  iree_tokenizer_normalizer_t** children = NULL;
  if (iree_status_is_ok(status) && json_child_count > 0) {
    status = iree_allocator_malloc_array(allocator, json_child_count,
                                         sizeof(*children), (void**)&children);
  }
  if (iree_status_is_ok(status) && children) {
    memset(children, 0, json_child_count * sizeof(*children));
  }

  // Parse each child normalizer using enumeration.
  iree_tokenizer_normalizer_sequence_parse_context_t context = {
      .allocator = allocator,
      .children = children,
      .child_count = json_child_count,
      .parsed_count = 0,
  };
  if (iree_status_is_ok(status) && json_child_count > 0) {
    status = iree_json_enumerate_array(
        normalizers_value,
        iree_tokenizer_parse_normalizer_sequence_child_visitor, &context);
  }

  // Compact array: filter out NULL children (no-op normalizers).
  iree_host_size_t non_null_count = 0;
  if (iree_status_is_ok(status) && children) {
    for (iree_host_size_t i = 0; i < json_child_count; ++i) {
      if (children[i]) {
        children[non_null_count++] = children[i];
      }
    }
  }

  // Dispatch based on child count after compaction:
  //   0 children → NULL (passthrough)
  //   1 child    → return the child directly (no sequence wrapper)
  //   2+ children → create sequence
  if (iree_status_is_ok(status)) {
    if (non_null_count == 0) {
      *out_normalizer = NULL;
    } else if (non_null_count == 1) {
      *out_normalizer = children[0];
      children[0] = NULL;  // Prevent cleanup from freeing it.
    } else {
      status = iree_tokenizer_normalizer_sequence_allocate(
          children, non_null_count, allocator, out_normalizer);
    }
  }

  // Cleanup on failure: free any successfully parsed children.
  if (!iree_status_is_ok(status) && children) {
    for (iree_host_size_t i = 0; i < json_child_count; ++i) {
      if (children[i]) {
        iree_tokenizer_normalizer_free(children[i]);
      }
    }
  }

  // Free the temporary array (sequence made its own copy for 2+ case).
  if (children) {
    iree_allocator_free(allocator, children);
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Simple Normalizers
//===----------------------------------------------------------------------===//

// Parses a Strip normalizer.
// JSON structure: {"type": "Strip", "strip_left": bool, "strip_right": bool}
// Both fields are required (HF Strip has no #[serde(default)] on fields).
static iree_status_t iree_tokenizer_parse_strip_normalizer(
    iree_string_view_t normalizer_value, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  *out_normalizer = NULL;
  bool strip_left = false;
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(
      normalizer_value, IREE_SV("strip_left"), &strip_left));
  bool strip_right = false;
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(
      normalizer_value, IREE_SV("strip_right"), &strip_right));
  return iree_tokenizer_normalizer_strip_allocate(strip_left, strip_right,
                                                  allocator, out_normalizer);
}

// Parses a Prepend normalizer.
// JSON structure: {"type": "Prepend", "prepend": "string"}
// The prepend field is required (HF Prepend has no #[serde(default)]).
// The value can be multi-byte UTF-8 like "▁" (U+2581).
static iree_status_t iree_tokenizer_parse_prepend_normalizer(
    iree_string_view_t normalizer_value, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  *out_normalizer = NULL;
  char prepend_buffer[IREE_TOKENIZER_NORMALIZER_PREPEND_MAX_LENGTH];
  iree_host_size_t prepend_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_lookup_string(
      normalizer_value, IREE_SV("prepend"),
      iree_make_mutable_string_view(prepend_buffer, sizeof(prepend_buffer)),
      &prepend_length));
  iree_string_view_t prepend_value =
      iree_make_string_view(prepend_buffer, prepend_length);
  return iree_tokenizer_normalizer_prepend_allocate(
      prepend_value, /*skip_if_prefix_matches=*/false, allocator,
      out_normalizer);
}

// Parses a StripAccents normalizer.
// JSON structure: {"type": "StripAccents"}
// No additional fields.
//
// This normalizer does not perform NFD decomposition. It only filters
// characters with the Unicode Mark category (Mn, Mc, Me).
// Precomposed characters like 'é' (U+00E9) pass through unchanged.
// To strip accents from precomposed text, chain NFD before StripAccents.
static iree_status_t iree_tokenizer_parse_strip_accents_normalizer(
    iree_string_view_t normalizer_value, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  *out_normalizer = NULL;
  // StripAccents has no configurable fields - validate only "type" is present.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      normalizer_value, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  return iree_tokenizer_normalizer_strip_accents_allocate(allocator,
                                                          out_normalizer);
}

//===----------------------------------------------------------------------===//
// BERT Normalizer
//===----------------------------------------------------------------------===//

// Parses a BertNormalizer.
// JSON structure:
//   {
//     "type": "BertNormalizer",
//     "clean_text": true,            // Optional, default true
//     "handle_chinese_chars": true,  // Optional, default true
//     "strip_accents": null,         // Optional: true, false, or null
//     "lowercase": true              // Optional, default true
//   }
//
// When strip_accents is null (the default), it takes the same value as
// lowercase. This matches HuggingFace behavior exactly.
static iree_status_t iree_tokenizer_parse_bert_normalizer(
    iree_string_view_t normalizer_value, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  *out_normalizer = NULL;
  iree_tokenizer_bert_normalizer_flags_t flags =
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_NONE;

  // All three boolean fields are required (HF BertNormalizer has no
  // #[serde(default)] on these fields).
  bool clean_text = false;
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(
      normalizer_value, IREE_SV("clean_text"), &clean_text));
  if (clean_text) {
    flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT;
  }

  bool handle_chinese_chars = false;
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(normalizer_value,
                                             IREE_SV("handle_chinese_chars"),
                                             &handle_chinese_chars));
  if (handle_chinese_chars) {
    flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS;
  }

  bool lowercase = false;
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(normalizer_value,
                                             IREE_SV("lowercase"), &lowercase));
  if (lowercase) {
    flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE;
  }

  // Parse strip_accents (default null, meaning same as lowercase).
  // HuggingFace serializes this as: true, false, or null.
  bool strip_accents = lowercase;
  iree_string_view_t strip_accents_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      normalizer_value, IREE_SV("strip_accents"), &strip_accents_value));
  if (strip_accents_value.size > 0) {
    if (iree_string_view_equal(strip_accents_value, IREE_SV("true"))) {
      strip_accents = true;
    } else if (iree_string_view_equal(strip_accents_value, IREE_SV("false"))) {
      strip_accents = false;
    }
    // "null" or missing: strip_accents stays as lowercase value.
  }
  if (strip_accents) {
    flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS;
  }

  return iree_tokenizer_normalizer_bert_allocate(flags, allocator,
                                                 out_normalizer);
}

//===----------------------------------------------------------------------===//
// Replace Normalizer
//===----------------------------------------------------------------------===//

// Maximum regex pattern length for Replace normalizer parsing.
// This is a parsing buffer limit, not a runtime constraint. HuggingFace regex
// patterns in Replace normalizers are typically short (e.g., "\s+" for
// whitespace normalization). 64 bytes provides room for most patterns.
// Note: Regex Replace is currently UNIMPLEMENTED; this limit only affects
// validation before the UNIMPLEMENTED error is returned.
#define IREE_TOKENIZER_REPLACE_MAX_REGEX_PATTERN 64

// Parses a Replace normalizer.
// JSON structure:
//   {
//     "type": "Replace",
//     "pattern": {"String": "..."} or {"Regex": "..."},
//     "content": "..."
//   }
//
// Only String patterns are currently supported; Regex patterns return
// IREE_STATUS_UNIMPLEMENTED.
static iree_status_t iree_tokenizer_parse_replace_normalizer(
    iree_string_view_t normalizer_value, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  *out_normalizer = NULL;
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("pattern"),
      IREE_SVL("content"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      normalizer_value, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Look up the pattern object.
  iree_string_view_t pattern_obj = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      normalizer_value, IREE_SV("pattern"), &pattern_obj));

  // Validate pattern object keys (only String or Regex allowed).
  static const iree_string_view_t kPatternAllowedKeys[] = {
      IREE_SVL("String"),
      IREE_SVL("Regex"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      pattern_obj, kPatternAllowedKeys, IREE_ARRAYSIZE(kPatternAllowedKeys)));

  // Check if "String" key exists (literal string replacement).
  // Use iree_json_try_lookup_object_value to detect presence (returns empty
  // string_view if key not found), then unescape the value separately.
  iree_string_view_t string_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pattern_obj, IREE_SV("String"), &string_value));
  if (string_value.size > 0) {
    // String key exists - unescape and validate.
    char pattern_buffer[IREE_TOKENIZER_REPLACE_MAX_PATTERN];
    iree_host_size_t pattern_length = 0;
    IREE_RETURN_IF_ERROR(iree_json_unescape_string(
        string_value, sizeof(pattern_buffer), pattern_buffer, &pattern_length));

    if (pattern_length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Replace normalizer String pattern cannot be "
          "empty; use content \"\" with a non-empty pattern "
          "to delete occurrences");
    }
    iree_string_view_t pattern =
        iree_make_string_view(pattern_buffer, pattern_length);

    // Parse the content field.
    char content_buffer[IREE_TOKENIZER_REPLACE_MAX_CONTENT];
    iree_host_size_t content_length = 0;
    IREE_RETURN_IF_ERROR(iree_json_lookup_string(
        normalizer_value, IREE_SV("content"),
        iree_make_mutable_string_view(content_buffer, sizeof(content_buffer)),
        &content_length));
    iree_string_view_t content =
        iree_make_string_view(content_buffer, content_length);

    return iree_tokenizer_normalizer_replace_allocate(
        pattern, content, allocator, out_normalizer);
  }

  // Check if "Regex" key exists.
  iree_string_view_t regex_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pattern_obj, IREE_SV("Regex"), &regex_value));
  if (regex_value.size > 0) {
    // Regex key exists - unescape and validate.
    char regex_buffer[IREE_TOKENIZER_REPLACE_MAX_REGEX_PATTERN];
    iree_host_size_t regex_length = 0;
    IREE_RETURN_IF_ERROR(iree_json_unescape_string(
        regex_value, sizeof(regex_buffer), regex_buffer, &regex_length));

    if (regex_length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Replace normalizer Regex pattern cannot be empty");
    }
    iree_string_view_t regex_pattern =
        iree_make_string_view(regex_buffer, regex_length);

    // Parse content.
    char content_buffer[IREE_TOKENIZER_REGEX_REPLACE_MAX_CONTENT];
    iree_host_size_t content_length = 0;
    IREE_RETURN_IF_ERROR(iree_json_lookup_string(
        normalizer_value, IREE_SV("content"),
        iree_make_mutable_string_view(content_buffer, sizeof(content_buffer)),
        &content_length));
    iree_string_view_t content =
        iree_make_string_view(content_buffer, content_length);

    return iree_tokenizer_normalizer_regex_replace_allocate(
        regex_pattern, content, allocator, out_normalizer);
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "Replace normalizer pattern must be either "
                          "{\"String\": \"...\"} or {\"Regex\": \"...\"}");
}

//===----------------------------------------------------------------------===//
// Top-Level Normalizer Dispatcher
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_huggingface_parse_normalizer(
    iree_string_view_t normalizer_value, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  // Handle null normalizer.
  if (iree_string_view_equal(normalizer_value, IREE_SV("null"))) {
    return iree_ok_status();
  }

  // Get normalizer type.
  iree_string_view_t type_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      normalizer_value, IREE_SV("type"), &type_value));

  // Dispatch based on type.
  if (iree_string_view_equal(type_value, IREE_SV("Sequence"))) {
    return iree_tokenizer_parse_sequence_normalizer(normalizer_value, allocator,
                                                    out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Lowercase"))) {
    return iree_tokenizer_normalizer_lowercase_allocate(allocator,
                                                        out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("NFC"))) {
    return iree_tokenizer_normalizer_nfc_allocate(allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("NFD"))) {
    return iree_tokenizer_normalizer_nfd_allocate(allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("NFKD"))) {
    return iree_tokenizer_normalizer_nfkd_allocate(allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Strip"))) {
    return iree_tokenizer_parse_strip_normalizer(normalizer_value, allocator,
                                                 out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Prepend"))) {
    return iree_tokenizer_parse_prepend_normalizer(normalizer_value, allocator,
                                                   out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("StripAccents"))) {
    return iree_tokenizer_parse_strip_accents_normalizer(
        normalizer_value, allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("BertNormalizer"))) {
    return iree_tokenizer_parse_bert_normalizer(normalizer_value, allocator,
                                                out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Precompiled"))) {
    return iree_tokenizer_parse_precompiled_normalizer(
        normalizer_value, allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Replace"))) {
    return iree_tokenizer_parse_replace_normalizer(normalizer_value, allocator,
                                                   out_normalizer);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "normalizer type not yet supported: '%.*s'",
                          (int)type_value.size, type_value.data);
}
