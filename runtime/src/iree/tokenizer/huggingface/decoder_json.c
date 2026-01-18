// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/decoder_json.h"

#include "iree/base/internal/json.h"

//===----------------------------------------------------------------------===//
// Decoder type parsers
//===----------------------------------------------------------------------===//

// Parses a WordPiece decoder:
// {
//   "type": "WordPiece",
//   "prefix": "##",
//   "cleanup": true
// }
static iree_status_t iree_tokenizer_parse_wordpiece_decoder(
    iree_string_view_t json, iree_tokenizer_decoder_t* out_decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("prefix"),
      IREE_SVL("cleanup"),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_validate_object_keys(json, kAllowedKeys,
                                         IREE_ARRAYSIZE(kAllowedKeys)));

  iree_string_view_t prefix = iree_string_view_empty();
  bool cleanup = true;

  // Parse prefix (optional, default "##").
  // An empty string "" is valid and disables prefix stripping.
  iree_string_view_t prefix_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(json, IREE_SV("prefix"),
                                            &prefix_value));
  if (prefix_value.size > 0 &&
      !iree_string_view_equal(prefix_value, IREE_SV("null"))) {
    prefix = prefix_value;
  } else if (prefix_value.size == 0 && prefix_value.data == NULL) {
    // Field not present: prefix remains NULL, will use default "##".
  } else {
    // Empty string or was reset - use empty prefix (no stripping).
    static const char empty_string[] = "";
    prefix = iree_make_string_view(empty_string, 0);
  }

  // Parse cleanup (optional, default true).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_bool(json, IREE_SV("cleanup"), true, &cleanup));

  iree_tokenizer_wordpiece_decoder_flags_t flags =
      cleanup ? IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_CLEANUP_SPACES
              : IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_DEFAULT;
  iree_tokenizer_decoder_initialize_wordpiece(prefix, flags, out_decoder);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Parses a Metaspace decoder:
// {
//   "type": "Metaspace",
//   "replacement": "▁"
// }
//
// add_prefix_space/prepend_scheme control whether to strip leading metaspace
// from the first token. Our decoder always strips it, which matches the
// default behavior (add_prefix_space: true / prepend_scheme: "always").
static iree_status_t iree_tokenizer_parse_metaspace_decoder(
    iree_string_view_t json, iree_tokenizer_decoder_t* out_decoder) {
  // Validate allowed keys.
  // Note: 'split' is an encoding-only parameter that doesn't affect decoding.
  // We accept it here because we fully support it in the encoder.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("replacement"),
      IREE_SVL("add_prefix_space"),
      IREE_SVL("prepend_scheme"),
      IREE_SVL("split"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  uint32_t replacement = IREE_TOKENIZER_METASPACE_REPLACEMENT;
  iree_tokenizer_metaspace_decoder_flags_t flags =
      IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING;  // Default: true.

  // Parse replacement (optional).
  iree_string_view_t replacement_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json, IREE_SV("replacement"), &replacement_value));
  if (replacement_value.size > 0 &&
      !iree_string_view_equal(replacement_value, IREE_SV("null"))) {
    IREE_RETURN_IF_ERROR(iree_json_parse_codepoint(
        replacement_value, IREE_TOKENIZER_METASPACE_REPLACEMENT, &replacement));
  }

  // Parse prepend_scheme (takes precedence over add_prefix_space).
  // "always" or "first" -> strip leading (add_prefix_space: true).
  // "never" -> don't strip (add_prefix_space: false).
  iree_string_view_t prepend_scheme_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json, IREE_SV("prepend_scheme"), &prepend_scheme_value));
  if (prepend_scheme_value.size > 0) {
    if (iree_string_view_equal(prepend_scheme_value, IREE_SV("never"))) {
      flags &= ~IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING;
    }
    // "always" and "first" both strip leading - keep default flag set.
  } else {
    // Fall back to add_prefix_space (legacy key).
    bool add_prefix_space = true;
    IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(
        json, IREE_SV("add_prefix_space"), true, &add_prefix_space));
    if (!add_prefix_space) {
      flags &= ~IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING;
    }
  }

  iree_tokenizer_decoder_initialize_metaspace(replacement, flags, out_decoder);
  return iree_ok_status();
}

// Parses a ByteLevel decoder: {"type": "ByteLevel", "add_prefix_space": true}
//
// trim_offsets is accepted but not used since we don't track character offsets.
// It only affects offset metadata, not decoded text.
//
// use_regex is an encoding-only parameter that controls whether to use regex
// splitting in the pre-tokenizer. We fully support it there, so we accept it
// here to allow parsing real-world configs (BART, Llama 3.x, CLIP, etc.).
static iree_status_t iree_tokenizer_parse_byte_level_decoder(
    iree_string_view_t json, iree_tokenizer_decoder_t* out_decoder) {
  // Validate allowed keys.
  // trim_offsets: accepted but unused (only affects offset metadata).
  // use_regex: encoding-only parameter, fully supported in pre-tokenizer.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("add_prefix_space"),
      IREE_SVL("trim_offsets"),
      IREE_SVL("use_regex"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Parse add_prefix_space (optional, default true to match HuggingFace).
  bool add_prefix_space = true;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(
      json, IREE_SV("add_prefix_space"), true, &add_prefix_space));
  iree_tokenizer_byte_level_decoder_flags_t flags =
      add_prefix_space ? IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_ADD_PREFIX_SPACE
                       : IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT;
  iree_tokenizer_decoder_initialize_byte_level(flags, out_decoder);
  return iree_ok_status();
}

// Parses a BPE decoder: {"type": "BPE"}
static iree_status_t iree_tokenizer_parse_bpe_decoder(
    iree_string_view_t json, iree_tokenizer_decoder_t* out_decoder) {
  // Validate allowed keys.
  // NOTE: suffix is NOT in this list because we don't implement it (the BPE
  // decoder just joins tokens). If a tokenizer needs suffix handling,
  // validation will fail.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_decoder_initialize_bpe(out_decoder);
  return iree_ok_status();
}

// Parses a Replace decoder:
// {
//   "type": "Replace",
//   "pattern": {"String": "▁"},
//   "content": " "
// }
static iree_status_t iree_tokenizer_parse_replace_decoder(
    iree_string_view_t json, iree_tokenizer_decoder_t* out_decoder) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("pattern"),
      IREE_SVL("content"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Parse pattern object.
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

  // Check for String pattern (the common case).
  iree_string_view_t string_pattern = iree_string_view_empty();
  iree_status_t status = iree_json_try_lookup_object_value(
      pattern_obj, IREE_SV("String"), &string_pattern);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }

  // If String pattern not found or empty, check for Regex pattern.
  if (string_pattern.size == 0) {
    iree_string_view_t regex_pattern = iree_string_view_empty();
    status = iree_json_try_lookup_object_value(pattern_obj, IREE_SV("Regex"),
                                               &regex_pattern);
    if (iree_status_is_ok(status) && regex_pattern.size > 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Regex patterns not yet supported in Replace decoder; "
          "only String patterns are currently implemented");
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Replace pattern must be String type");
  }

  // Unescape the pattern string.
  char pattern_buffer[IREE_TOKENIZER_REPLACE_DECODER_PATTERN_MAX_SIZE];
  iree_host_size_t pattern_length = 0;
  status = iree_json_unescape_string(string_pattern, sizeof(pattern_buffer),
                                     pattern_buffer, &pattern_length);
  if (!iree_status_is_ok(status)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Replace pattern too long (max %zu bytes)",
                            sizeof(pattern_buffer));
  }

  // Parse content.
  iree_string_view_t content_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("content"), &content_value));

  // Unescape the content string.
  char content_buffer[IREE_TOKENIZER_REPLACE_DECODER_CONTENT_MAX_SIZE];
  iree_host_size_t content_length = 0;
  status = iree_json_unescape_string(content_value, sizeof(content_buffer),
                                     content_buffer, &content_length);
  if (!iree_status_is_ok(status)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Replace content too long (max %zu bytes)",
                            sizeof(content_buffer));
  }

  return iree_tokenizer_decoder_initialize_replace(
      iree_make_string_view(pattern_buffer, pattern_length),
      iree_make_string_view(content_buffer, content_length), out_decoder);
}

// Parses a ByteFallback decoder: {"type": "ByteFallback"}
static iree_status_t iree_tokenizer_parse_byte_fallback_decoder(
    iree_string_view_t json, iree_tokenizer_decoder_t* out_decoder) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_decoder_initialize_byte_fallback(out_decoder);
  return iree_ok_status();
}

// Parses a Fuse decoder: {"type": "Fuse"}
static iree_status_t iree_tokenizer_parse_fuse_decoder(
    iree_string_view_t json, iree_tokenizer_decoder_t* out_decoder) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_decoder_initialize_fuse(out_decoder);
  return iree_ok_status();
}

// Parses a Strip decoder:
// {
//   "type": "Strip",
//   "content": " ",
//   "start": 1,
//   "stop": 0
// }
static iree_status_t iree_tokenizer_parse_strip_decoder(
    iree_string_view_t json, iree_tokenizer_decoder_t* out_decoder) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("content"),
      IREE_SVL("start"),
      IREE_SVL("stop"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Parse content.
  iree_string_view_t content_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("content"), &content_value));

  // Unescape the content string.
  char content_buffer[IREE_TOKENIZER_STRIP_DECODER_CONTENT_MAX_SIZE];
  iree_host_size_t content_length = 0;
  iree_status_t status = iree_json_unescape_string(
      content_value, sizeof(content_buffer), content_buffer, &content_length);
  if (!iree_status_is_ok(status)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strip content too long (max %zu bytes)",
                            sizeof(content_buffer));
  }

  // Parse start (optional, default 0).
  int64_t start_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_int64(json, IREE_SV("start"), 0, &start_value));

  // Parse stop (optional, default 0).
  int64_t stop_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_int64(json, IREE_SV("stop"), 0, &stop_value));

  return iree_tokenizer_decoder_initialize_strip(
      iree_make_string_view(content_buffer, content_length),
      (uint8_t)(start_value > 255 ? 255 : start_value),
      (uint8_t)(stop_value > 255 ? 255 : stop_value), out_decoder);
}

// Forward declaration for recursive parsing.
static iree_status_t iree_tokenizer_decoder_parse_decoder_field_with_allocator(
    iree_string_view_t decoder_json, iree_allocator_t allocator,
    iree_tokenizer_decoder_t* out_decoder);

// Parses a Sequence decoder:
// {
//   "type": "Sequence",
//   "decoders": [...]
// }
static iree_status_t iree_tokenizer_parse_sequence_decoder(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_decoder_t* out_decoder) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("decoders"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Get decoders array.
  iree_string_view_t decoders_array = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(json, IREE_SV("decoders"),
                                                     &decoders_array));

  // Count decoders.
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(decoders_array, &count));

  if (count == 0) {
    iree_tokenizer_decoder_initialize_none(out_decoder);
    return iree_ok_status();
  }

  // Check for overflow in children array size.
  if (count > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_decoder_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence child count overflow");
  }
  iree_host_size_t children_size = count * sizeof(iree_tokenizer_decoder_t);

  // Allocate children array.
  iree_tokenizer_decoder_t* children = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, children_size, (void**)&children));
  memset(children, 0, children_size);

  // Set up structure immediately so deinitialize handles cleanup on any error.
  // Note: decoder's initialize_sequence takes direct ownership, so set up here.
  out_decoder->type = IREE_TOKENIZER_DECODER_SEQUENCE;
  out_decoder->config.sequence.count = count;
  out_decoder->config.sequence.children = children;
  out_decoder->config.sequence.allocator = allocator;

  // Parse each child decoder.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_string_view_t child_json = iree_string_view_empty();
    status = iree_json_array_get(decoders_array, i, &child_json);
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_decoder_parse_decoder_field_with_allocator(
          child_json, allocator, &children[i]);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_tokenizer_decoder_deinitialize(out_decoder);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Internal: parse decoder field with explicit allocator for Sequence support.
static iree_status_t iree_tokenizer_decoder_parse_decoder_field_with_allocator(
    iree_string_view_t decoder_json, iree_allocator_t allocator,
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_initialize_none(out_decoder);

  // Look up the type field.
  iree_string_view_t type_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(decoder_json, IREE_SV("type"),
                                        &type_value));

  // Dispatch based on type.
  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(type_value, IREE_SV("WordPiece"))) {
    status = iree_tokenizer_parse_wordpiece_decoder(decoder_json, out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Metaspace"))) {
    status = iree_tokenizer_parse_metaspace_decoder(decoder_json, out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("ByteLevel"))) {
    status = iree_tokenizer_parse_byte_level_decoder(decoder_json, out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("BPE"))) {
    status = iree_tokenizer_parse_bpe_decoder(decoder_json, out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Sequence"))) {
    status = iree_tokenizer_parse_sequence_decoder(decoder_json, allocator,
                                                   out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Replace"))) {
    status = iree_tokenizer_parse_replace_decoder(decoder_json, out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("ByteFallback"))) {
    status =
        iree_tokenizer_parse_byte_fallback_decoder(decoder_json, out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Fuse"))) {
    status = iree_tokenizer_parse_fuse_decoder(decoder_json, out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Strip"))) {
    status = iree_tokenizer_parse_strip_decoder(decoder_json, out_decoder);
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported decoder type '%.*s'",
                              (int)type_value.size, type_value.data);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_decoder_parse_decoder_field(
    iree_string_view_t decoder_json, iree_tokenizer_decoder_t* out_decoder) {
  // Use system allocator for Sequence decoder children.
  return iree_tokenizer_decoder_parse_decoder_field_with_allocator(
      decoder_json, iree_allocator_system(), out_decoder);
}

iree_status_t iree_tokenizer_decoder_parse_json(
    iree_string_view_t json_root, iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_initialize_none(out_decoder);

  // Look up the decoder field (optional).
  iree_string_view_t decoder_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(json_root, IREE_SV("decoder"),
                                            &decoder_value));

  if (decoder_value.size == 0 ||
      iree_string_view_equal(decoder_value, IREE_SV("null"))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status =
      iree_tokenizer_decoder_parse_decoder_field(decoder_value, out_decoder);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
