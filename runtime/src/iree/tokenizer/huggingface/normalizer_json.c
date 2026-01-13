// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/normalizer_json.h"

#include "iree/base/internal/json.h"
#include "iree/tokenizer/regex/compile.h"

//===----------------------------------------------------------------------===//
// Normalizer type parsers
//===----------------------------------------------------------------------===//

// Parses a BertNormalizer:
// {
//   "type": "BertNormalizer",
//   "clean_text": true,
//   "handle_chinese_chars": true,
//   "strip_accents": null,
//   "lowercase": true
// }
static iree_status_t iree_tokenizer_parse_bert_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("clean_text"),
      IREE_SVL("handle_chinese_chars"),
      IREE_SVL("strip_accents"),
      IREE_SVL("lowercase"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_bert_normalizer_flags_t flags =
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT;

  bool clean_text = true;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(json, IREE_SV("clean_text"),
                                                 true, &clean_text));
  if (clean_text) {
    flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT;
  }

  bool handle_chinese_chars = true;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(
      json, IREE_SV("handle_chinese_chars"), true, &handle_chinese_chars));
  if (handle_chinese_chars) {
    flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS;
  }

  // strip_accents defaults to null (meaning: use lowercase value).
  iree_string_view_t strip_accents_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json, IREE_SV("strip_accents"), &strip_accents_value));

  bool lowercase = true;
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_bool(json, IREE_SV("lowercase"), true, &lowercase));
  if (lowercase) {
    flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE;
  }

  // Handle strip_accents: if null, use lowercase value.
  if (strip_accents_value.size == 0 ||
      iree_string_view_equal(strip_accents_value, IREE_SV("null"))) {
    if (lowercase) {
      flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS;
    }
  } else {
    bool strip_accents = false;
    IREE_RETURN_IF_ERROR(
        iree_json_parse_bool(strip_accents_value, &strip_accents));
    if (strip_accents) {
      flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS;
    }
  }

  iree_tokenizer_normalizer_initialize_bert(flags, out_normalizer);
  return iree_ok_status();
}

// Parses a Lowercase: {"type": "Lowercase"}
static iree_status_t iree_tokenizer_parse_lowercase_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_normalizer_initialize_lowercase(out_normalizer);
  return iree_ok_status();
}

// Parses a StripAccents: {"type": "StripAccents"}
static iree_status_t iree_tokenizer_parse_strip_accents_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_normalizer_initialize_strip_accents(out_normalizer);
  return iree_ok_status();
}

// Parses an NFD: {"type": "NFD"}
// NFD (Canonical Decomposition) is equivalent to StripAccents for our purposes.
static iree_status_t iree_tokenizer_parse_nfd_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_normalizer_initialize_strip_accents(out_normalizer);
  return iree_ok_status();
}

// Parses an NFC: {"type": "NFC"}
static iree_status_t iree_tokenizer_parse_nfc_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_normalizer_initialize_nfc(out_normalizer);
  return iree_ok_status();
}

// Parses a Strip: {"type": "Strip", "strip_left": true, "strip_right": true}
static iree_status_t iree_tokenizer_parse_strip_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("strip_left"),
      IREE_SVL("strip_right"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Parse strip_left (optional, default true to match HuggingFace).
  bool strip_left = true;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(json, IREE_SV("strip_left"),
                                                 true, &strip_left));

  // Parse strip_right (optional, default true to match HuggingFace).
  bool strip_right = true;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(json, IREE_SV("strip_right"),
                                                 true, &strip_right));

  iree_tokenizer_normalizer_initialize_strip(strip_left, strip_right,
                                             out_normalizer);
  return iree_ok_status();
}

// Parses a Prepend: {"type": "Prepend", "prepend": "▁"}
static iree_status_t iree_tokenizer_parse_prepend_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("prepend"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_string_view_t prepend_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("prepend"), &prepend_value));

  // Unescape the string value (handles \u2581 etc).
  char prepend_buffer[IREE_TOKENIZER_PREPEND_NORMALIZER_MAX_SIZE];
  iree_host_size_t prepend_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(
      prepend_value, sizeof(prepend_buffer), prepend_buffer, &prepend_length));

  return iree_tokenizer_normalizer_initialize_prepend(
      iree_make_string_view(prepend_buffer, prepend_length), out_normalizer);
}

// Parses a Replace: {"type": "Replace", "pattern": {"String": " "}, "content":
// "▁"} or {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "}
static iree_status_t iree_tokenizer_parse_replace_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("pattern"),
      IREE_SVL("content"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Look up pattern object: {"String": "..."} or {"Regex": "..."}
  iree_string_view_t pattern_obj = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("pattern"), &pattern_obj));

  // Validate pattern object keys.
  static const iree_string_view_t kPatternAllowedKeys[] = {
      IREE_SVL("String"),
      IREE_SVL("Regex"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      pattern_obj, kPatternAllowedKeys, IREE_ARRAYSIZE(kPatternAllowedKeys)));

  // Try to get String pattern first.
  iree_string_view_t string_pattern = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pattern_obj, IREE_SV("String"), &string_pattern));

  // Look up content (needed for both String and Regex).
  iree_string_view_t content_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("content"), &content_value));

  // Unescape content (used by both paths).
  // Buffer sized to match IREE_TOKENIZER_REPLACE_NORMALIZER_CONTENT_MAX_SIZE.
  char content_buffer[256];
  iree_host_size_t content_length = 0;
  iree_status_t unescape_status = iree_json_unescape_string(
      content_value, sizeof(content_buffer), content_buffer, &content_length);
  if (!iree_status_is_ok(unescape_status)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Replace content too long (max %zu bytes)",
                            sizeof(content_buffer));
  }

  // If String pattern not found or empty, check for Regex pattern.
  if (string_pattern.size == 0) {
    iree_string_view_t regex_pattern = iree_string_view_empty();
    iree_status_t status = iree_json_try_lookup_object_value(
        pattern_obj, IREE_SV("Regex"), &regex_pattern);
    if (iree_status_is_ok(status) && regex_pattern.size > 0) {
      // Unescape the regex pattern.
      // Buffer sized for typical tokenizer patterns; complex Unicode patterns
      // may need more space.
      char regex_buffer[1024];
      iree_host_size_t regex_length = 0;
      iree_status_t regex_unescape_status = iree_json_unescape_string(
          regex_pattern, sizeof(regex_buffer), regex_buffer, &regex_length);
      if (!iree_status_is_ok(regex_unescape_status)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "ReplaceRegex pattern too long (max %zu bytes)",
                                sizeof(regex_buffer));
      }

      // Compile the regex pattern.
      iree_tokenizer_regex_compile_error_t compile_error = {0};
      uint8_t* dfa_data = NULL;
      iree_host_size_t dfa_size = 0;
      status = iree_tokenizer_regex_compile(
          iree_make_string_view(regex_buffer, regex_length),
          IREE_TOKENIZER_REGEX_COMPILE_FLAG_NONE, allocator, &dfa_data,
          &dfa_size, &compile_error);
      if (!iree_status_is_ok(status)) {
        return iree_status_annotate_f(
            status, "failed to compile regex pattern at position %zu: %s",
            compile_error.position,
            compile_error.message ? compile_error.message : "unknown error");
      }

      // Initialize the replace-regex normalizer.
      status = iree_tokenizer_normalizer_initialize_replace_regex(
          dfa_data, dfa_size,
          iree_make_string_view(content_buffer, content_length), allocator,
          out_normalizer);
      if (!iree_status_is_ok(status)) {
        iree_allocator_free(allocator, dfa_data);
        return status;
      }
      return iree_ok_status();
    }
  }

  // String pattern path.
  char pattern_buffer[IREE_TOKENIZER_REPLACE_NORMALIZER_PATTERN_MAX_SIZE];
  iree_host_size_t pattern_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(
      string_pattern, sizeof(pattern_buffer), pattern_buffer, &pattern_length));

  return iree_tokenizer_normalizer_initialize_replace(
      iree_make_string_view(pattern_buffer, pattern_length),
      iree_make_string_view(content_buffer, content_length), out_normalizer);
}

//===----------------------------------------------------------------------===//
// Base64 Decoding
//===----------------------------------------------------------------------===//

// Base64 decoding table (6 bits per character).
// -1 = invalid, -2 = padding ('=').
static const int8_t iree_base64_decode_table[256] = {
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
// Returns the number of bytes written to |out_data|.
// Returns error if the input is malformed or buffer is too small.
static iree_status_t iree_base64_decode(iree_string_view_t encoded,
                                        uint8_t* out_data,
                                        iree_host_size_t out_capacity,
                                        iree_host_size_t* out_length) {
  // Calculate required output size.
  // Base64 uses 4 characters to encode 3 bytes.
  // Padding ('=') at end reduces output by 1 or 2 bytes.
  iree_host_size_t padding = 0;
  if (encoded.size > 0 && encoded.data[encoded.size - 1] == '=') padding++;
  if (encoded.size > 1 && encoded.data[encoded.size - 2] == '=') padding++;
  // Guard against underflow: padding can never exceed the computed base size.
  iree_host_size_t base_decoded = (encoded.size / 4) * 3;
  if (padding > base_decoded) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid base64 padding");
  }
  iree_host_size_t decoded_size = base_decoded - padding;

  if (out_capacity < decoded_size) {
    *out_length = decoded_size;
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "base64 output buffer too small: need %zu, have %zu", decoded_size,
        out_capacity);
  }

  iree_host_size_t out_position = 0;
  uint32_t accumulator = 0;
  int bits = 0;

  for (iree_host_size_t i = 0; i < encoded.size; ++i) {
    uint8_t c = (uint8_t)encoded.data[i];
    int8_t value = iree_base64_decode_table[c];
    if (value == -1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid base64 character at position %zu", i);
    }
    if (value == -2) {
      // Padding character - ignore.
      continue;
    }
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
// Precompiled Normalizer Parser
//===----------------------------------------------------------------------===//

// Parses a Precompiled normalizer (SentencePiece):
// {
//   "type": "Precompiled",
//   "precompiled_charsmap": "base64-encoded-data"
// }
static iree_status_t iree_tokenizer_parse_precompiled_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("precompiled_charsmap"),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_validate_object_keys(json, kAllowedKeys,
                                         IREE_ARRAYSIZE(kAllowedKeys)));

  // Look up the precompiled_charsmap field.
  iree_string_view_t charsmap_b64 = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(json, IREE_SV("precompiled_charsmap"),
                                        &charsmap_b64));

  // Handle empty charsmap (identity normalizer).
  if (charsmap_b64.size == 0) {
    iree_tokenizer_normalizer_initialize_none(out_normalizer);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Calculate decoded size.
  // Guard against underflow: padding can never exceed the computed base size.
  iree_host_size_t padding = 0;
  if (charsmap_b64.size > 0 && charsmap_b64.data[charsmap_b64.size - 1] == '=')
    padding++;
  if (charsmap_b64.size > 1 && charsmap_b64.data[charsmap_b64.size - 2] == '=')
    padding++;
  iree_host_size_t base_decoded = (charsmap_b64.size / 4) * 3;
  if (padding > base_decoded) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid base64 padding in precompiled charsmap");
  }
  iree_host_size_t decoded_size = base_decoded - padding;

  // Allocate temporary buffer for decoded data.
  uint8_t* decoded_data = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, decoded_size, (void**)&decoded_data);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Decode base64.
  iree_host_size_t actual_length = 0;
  status = iree_base64_decode(charsmap_b64, decoded_data, decoded_size,
                              &actual_length);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, decoded_data);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Initialize the precompiled normalizer from decoded data.
  status = iree_tokenizer_normalizer_initialize_precompiled(
      decoded_data, actual_length, allocator, out_normalizer);

  // Free temporary buffer.
  iree_allocator_free(allocator, decoded_data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Sequence Normalizer Parser
//===----------------------------------------------------------------------===//

// Parses a Sequence: {"type": "Sequence", "normalizers": [...]}
static iree_status_t iree_tokenizer_parse_sequence_normalizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("normalizers"),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_validate_object_keys(json, kAllowedKeys,
                                         IREE_ARRAYSIZE(kAllowedKeys)));

  // Look up the normalizers array.
  iree_string_view_t normalizers_array = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(json, IREE_SV("normalizers"),
                                        &normalizers_array));

  // Count elements.
  iree_host_size_t count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_array_length(normalizers_array, &count));

  if (count == 0) {
    iree_tokenizer_normalizer_initialize_none(out_normalizer);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Check for overflow in children array size.
  if (count > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_normalizer_t)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence child count overflow");
  }
  iree_host_size_t children_size = count * sizeof(iree_tokenizer_normalizer_t);

  // Allocate children array.
  iree_tokenizer_normalizer_t* children = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, children_size, (void**)&children));
  memset(children, 0, children_size);

  // Set up structure immediately so deinitialize handles cleanup on any error.
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_SEQUENCE;
  out_normalizer->config.sequence.count = count;
  out_normalizer->config.sequence.children = children;
  out_normalizer->config.sequence.allocator = allocator;

  // Parse each child normalizer.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_string_view_t child_json = iree_string_view_empty();
    status = iree_json_array_get(normalizers_array, i, &child_json);
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_normalizer_parse_normalizer_field(
          child_json, allocator, &children[i]);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_tokenizer_normalizer_deinitialize(out_normalizer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_normalizer_parse_normalizer_field(
    iree_string_view_t normalizer_json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  IREE_ASSERT_ARGUMENT(out_normalizer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_initialize_none(out_normalizer);

  // Look up the type field.
  iree_string_view_t type_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(normalizer_json, IREE_SV("type"),
                                        &type_value));

  // Dispatch based on type.
  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(type_value, IREE_SV("BertNormalizer"))) {
    status = iree_tokenizer_parse_bert_normalizer(normalizer_json, allocator,
                                                  out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Lowercase"))) {
    status = iree_tokenizer_parse_lowercase_normalizer(
        normalizer_json, allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("StripAccents"))) {
    status = iree_tokenizer_parse_strip_accents_normalizer(
        normalizer_json, allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("NFD"))) {
    status = iree_tokenizer_parse_nfd_normalizer(normalizer_json, allocator,
                                                 out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Prepend"))) {
    status = iree_tokenizer_parse_prepend_normalizer(normalizer_json, allocator,
                                                     out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Replace"))) {
    status = iree_tokenizer_parse_replace_normalizer(normalizer_json, allocator,
                                                     out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Sequence"))) {
    status = iree_tokenizer_parse_sequence_normalizer(
        normalizer_json, allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("NFC"))) {
    status = iree_tokenizer_parse_nfc_normalizer(normalizer_json, allocator,
                                                 out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Precompiled"))) {
    status = iree_tokenizer_parse_precompiled_normalizer(
        normalizer_json, allocator, out_normalizer);
  } else if (iree_string_view_equal(type_value, IREE_SV("Strip"))) {
    status = iree_tokenizer_parse_strip_normalizer(normalizer_json, allocator,
                                                   out_normalizer);
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported normalizer type '%.*s'",
                              (int)type_value.size, type_value.data);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_normalizer_parse_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer) {
  IREE_ASSERT_ARGUMENT(out_normalizer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_initialize_none(out_normalizer);

  // Look up the normalizer field (optional).
  iree_string_view_t normalizer_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(json_root, IREE_SV("normalizer"),
                                            &normalizer_value));

  if (normalizer_value.size == 0 ||
      iree_string_view_equal(normalizer_value, IREE_SV("null"))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status = iree_tokenizer_normalizer_parse_normalizer_field(
      normalizer_value, allocator, out_normalizer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
