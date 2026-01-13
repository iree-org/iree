// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/transform_json.h"

#include "iree/base/internal/json.h"
#include "iree/tokenizer/huggingface/normalizer_json.h"

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// GPT-2 regex pattern for ByteLevel pretokenizer when use_regex=true.
// This splits on word boundaries while preserving the matched segments.
// Pattern: contractions | optional-space + letters | optional-space + numbers |
//          optional-space + other | trailing whitespace (negative lookahead) |
//          whitespace
static const char IREE_TOKENIZER_GPT2_REGEX_PATTERN[] =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
    "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Parses the "prepend_scheme" field from Metaspace config.
// Valid values: "always", "first", "never"
static iree_status_t iree_json_parse_prepend_scheme(
    iree_string_view_t value, iree_tokenizer_prepend_scheme_t* out_scheme) {
  if (iree_string_view_equal(value, IREE_SV("always"))) {
    *out_scheme = IREE_TOKENIZER_PREPEND_ALWAYS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("first"))) {
    *out_scheme = IREE_TOKENIZER_PREPEND_FIRST;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("never"))) {
    *out_scheme = IREE_TOKENIZER_PREPEND_NEVER;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown prepend_scheme '%.*s'", (int)value.size,
                          value.data);
}

//===----------------------------------------------------------------------===//
// Transform type parsers
//===----------------------------------------------------------------------===//

// Parses a BertPreTokenizer: {"type": "BertPreTokenizer"}
static iree_status_t iree_tokenizer_parse_bert_pretokenizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_text_transform_initialize_bert(out_transform);
  return iree_ok_status();
}

// Parses a Whitespace: {"type": "Whitespace"}
static iree_status_t iree_tokenizer_parse_whitespace_pretokenizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_text_transform_initialize_whitespace(out_transform);
  return iree_ok_status();
}

// Parses a ByteLevel: {"type": "ByteLevel", "add_prefix_space": true, ...}
//
// When use_regex=true (default), this creates a Sequence[Split(GPT2),
// ByteLevel] to match HuggingFace behavior. The GPT-2 regex splits on word
// boundaries.
//
// trim_offsets is parsed but ignored since we don't track character offsets.
static iree_status_t iree_tokenizer_parse_byte_level_pretokenizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("add_prefix_space"),
      IREE_SVL("trim_offsets"),
      IREE_SVL("use_regex"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_byte_level_flags_t flags =
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT;

  // Parse add_prefix_space (optional, default false).
  bool add_prefix_space = false;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(
      json, IREE_SV("add_prefix_space"), false, &add_prefix_space));
  if (add_prefix_space) {
    flags |= IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE;
  }

  // Parse use_regex (optional, default true to match HuggingFace).
  bool use_regex = true;
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_bool(json, IREE_SV("use_regex"), true, &use_regex));

  // trim_offsets is intentionally not parsed - we don't track offsets.

  if (!use_regex) {
    // Simple case: just ByteLevel encoding without regex splitting.
    iree_tokenizer_text_transform_initialize_byte_level(flags, out_transform);
    return iree_ok_status();
  }

  // use_regex=true: Create Sequence[Split(GPT2), ByteLevel].
  // This matches HuggingFace's behavior where ByteLevel uses the GPT-2 regex
  // pattern to split text on word boundaries before byte-level encoding.

  // Create both child transforms on the stack.
  iree_tokenizer_text_transform_t children[2];
  memset(children, 0, sizeof(children));

  // First child: Split with GPT-2 pattern.
  // behavior=1 (ISOLATED) emits matches as separate segments.
  // invert=false means we emit matches (the words), not gaps.
  iree_status_t status = iree_tokenizer_text_transform_initialize_split(
      iree_make_string_view(IREE_TOKENIZER_GPT2_REGEX_PATTERN,
                            sizeof(IREE_TOKENIZER_GPT2_REGEX_PATTERN) - 1),
      /*behavior=*/1,  // ISOLATED
      /*invert=*/false, allocator, &children[0]);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Second child: ByteLevel encoding.
  iree_tokenizer_text_transform_initialize_byte_level(flags, &children[1]);

  // Create the Sequence transform (takes ownership of children).
  status = iree_tokenizer_text_transform_initialize_sequence(
      children, IREE_ARRAYSIZE(children), allocator, out_transform);
  if (!iree_status_is_ok(status)) {
    // Clean up on failure.
    iree_tokenizer_text_transform_deinitialize(&children[0]);
    iree_tokenizer_text_transform_deinitialize(&children[1]);
    return status;
  }

  return iree_ok_status();
}

// Parses a Metaspace: {"type": "Metaspace", "replacement": "▁", ...}
//
// add_prefix_space is the legacy key for prepend_scheme:
//   add_prefix_space: true  -> prepend_scheme: "always"
//   add_prefix_space: false -> prepend_scheme: "never"
// If both are present, prepend_scheme takes precedence.
static iree_status_t iree_tokenizer_parse_metaspace_pretokenizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  (void)allocator;

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),           IREE_SVL("replacement"),
      IREE_SVL("prepend_scheme"), IREE_SVL("add_prefix_space"),
      IREE_SVL("split"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  uint32_t replacement = IREE_TOKENIZER_METASPACE_REPLACEMENT;
  iree_tokenizer_prepend_scheme_t prepend_scheme =
      IREE_TOKENIZER_PREPEND_ALWAYS;
  iree_tokenizer_metaspace_flags_t flags =
      IREE_TOKENIZER_METASPACE_FLAG_DEFAULT;

  // Parse replacement (optional).
  iree_string_view_t replacement_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json, IREE_SV("replacement"), &replacement_value));
  if (replacement_value.size > 0) {
    IREE_RETURN_IF_ERROR(iree_json_parse_codepoint(
        replacement_value, IREE_TOKENIZER_METASPACE_REPLACEMENT, &replacement));
  }

  // Parse prepend_scheme (optional, takes precedence over add_prefix_space).
  iree_string_view_t prepend_scheme_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json, IREE_SV("prepend_scheme"), &prepend_scheme_value));
  if (prepend_scheme_value.size > 0) {
    IREE_RETURN_IF_ERROR(
        iree_json_parse_prepend_scheme(prepend_scheme_value, &prepend_scheme));
  } else {
    // Fall back to add_prefix_space (legacy key).
    bool add_prefix_space = true;  // Default matches prepend_scheme: "always".
    IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(
        json, IREE_SV("add_prefix_space"), true, &add_prefix_space));
    prepend_scheme = add_prefix_space ? IREE_TOKENIZER_PREPEND_ALWAYS
                                      : IREE_TOKENIZER_PREPEND_NEVER;
  }

  // Parse split (optional, HuggingFace uses "split" for the split flag).
  iree_string_view_t split_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_object_value(json, IREE_SV("split"), &split_value));
  if (split_value.size > 0) {
    bool split = false;
    IREE_RETURN_IF_ERROR(iree_json_parse_bool(split_value, &split));
    if (split) {
      flags |= IREE_TOKENIZER_METASPACE_FLAG_SPLIT;
    }
  }

  iree_tokenizer_text_transform_initialize_metaspace(
      replacement, prepend_scheme, flags, out_transform);
  return iree_ok_status();
}

// Parses the "behavior" field from Split config.
// Valid values: "Removed", "Isolated", "MergedWithPrevious", "MergedWithNext",
// "Contiguous"
static iree_status_t iree_json_parse_split_behavior(
    iree_string_view_t value,
    iree_tokenizer_regex_split_behavior_t* out_behavior) {
  if (iree_string_view_equal(value, IREE_SV("Removed"))) {
    *out_behavior = IREE_TOKENIZER_REGEX_SPLIT_REMOVED;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("Isolated"))) {
    *out_behavior = IREE_TOKENIZER_REGEX_SPLIT_ISOLATED;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("MergedWithPrevious"))) {
    *out_behavior = IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_PREVIOUS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("MergedWithNext"))) {
    *out_behavior = IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("Contiguous"))) {
    *out_behavior = IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown split behavior '%.*s'", (int)value.size,
                          value.data);
}

// Parses a Split: {"type": "Split", "pattern": {"Regex": "..."}, ...}
// HuggingFace format:
//   "pattern": {"Regex": "..."}  or {"String": "literal"}
//   "behavior": "Removed" | "Isolated" | "MergedWithPrevious" | etc.
//   "invert": true | false
static iree_status_t iree_tokenizer_parse_split_pretokenizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("pattern"),
      IREE_SVL("behavior"),
      IREE_SVL("invert"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_regex_split_behavior_t behavior =
      IREE_TOKENIZER_REGEX_SPLIT_REMOVED;
  bool invert = false;

  // Parse pattern (required).
  iree_string_view_t pattern_object = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("pattern"), &pattern_object));

  // Validate pattern object keys.
  static const iree_string_view_t kPatternAllowedKeys[] = {
      IREE_SVL("Regex"),
      IREE_SVL("String"),
  };
  IREE_RETURN_IF_ERROR(
      iree_json_validate_object_keys(pattern_object, kPatternAllowedKeys,
                                     IREE_ARRAYSIZE(kPatternAllowedKeys)));

  // Extract regex pattern from {"Regex": "..."} or {"String": "..."}
  // First try Regex, then String.
  iree_string_view_t json_string_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pattern_object, IREE_SV("Regex"), &json_string_value));
  if (json_string_value.size == 0) {
    IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
        pattern_object, IREE_SV("String"), &json_string_value));
  }
  if (json_string_value.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Split pattern must have 'Regex' or 'String'");
  }

  // Unescape the JSON string value into a buffer.
  // Buffer sized for typical tokenizer patterns; complex Unicode patterns
  // (e.g., Llama 3's regex) may need more space.
  char pattern_buffer[2048];
  iree_host_size_t pattern_length = 0;
  iree_status_t pattern_status =
      iree_json_unescape_string(json_string_value, sizeof(pattern_buffer),
                                pattern_buffer, &pattern_length);
  if (!iree_status_is_ok(pattern_status)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Split pattern too long (max %zu bytes)",
                            sizeof(pattern_buffer));
  }
  iree_string_view_t pattern_string =
      iree_make_string_view(pattern_buffer, pattern_length);

  // Parse behavior (optional, default "Removed").
  iree_string_view_t behavior_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json, IREE_SV("behavior"), &behavior_value));
  if (behavior_value.size > 0) {
    IREE_RETURN_IF_ERROR(
        iree_json_parse_split_behavior(behavior_value, &behavior));
  }

  // Parse invert (optional, default false).
  iree_string_view_t invert_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json, IREE_SV("invert"), &invert_value));
  if (invert_value.size > 0) {
    IREE_RETURN_IF_ERROR(iree_json_parse_bool(invert_value, &invert));
  }

  return iree_tokenizer_text_transform_initialize_split(
      pattern_string, behavior, invert, allocator, out_transform);
}

// Parses a Sequence: {"type": "Sequence", "pretokenizers": [...]}
static iree_status_t iree_tokenizer_parse_sequence_pretokenizer(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("pretokenizers"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Look up the pretokenizers array.
  iree_string_view_t pretokenizers_array = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      json, IREE_SV("pretokenizers"), &pretokenizers_array));

  // Count elements.
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(pretokenizers_array, &count));

  if (count == 0) {
    iree_tokenizer_text_transform_initialize_none(out_transform);
    return iree_ok_status();
  }

  // Check for overflow in children array size.
  if (count > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_text_transform_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence child count overflow");
  }
  iree_host_size_t children_size =
      count * sizeof(iree_tokenizer_text_transform_t);

  // Allocate children array.
  iree_tokenizer_text_transform_t* children = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, children_size, (void**)&children));
  memset(children, 0, children_size);

  // Set up structure immediately so deinitialize handles cleanup on any error.
  out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE;
  out_transform->config.sequence.count = count;
  out_transform->config.sequence.children = children;
  out_transform->config.sequence.allocator = allocator;

  // Parse each child pretokenizer.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_string_view_t child_json = iree_string_view_empty();
    status = iree_json_array_get(pretokenizers_array, i, &child_json);
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_text_transform_parse_pretokenizer(
          child_json, allocator, &children[i]);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_tokenizer_text_transform_deinitialize(out_transform);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_text_transform_parse_pretokenizer(
    iree_string_view_t pretokenizer_json, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  IREE_ASSERT_ARGUMENT(out_transform);
  iree_tokenizer_text_transform_initialize_none(out_transform);

  // Look up the type field.
  iree_string_view_t type_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      pretokenizer_json, IREE_SV("type"), &type_value));

  // Dispatch based on type.
  if (iree_string_view_equal(type_value, IREE_SV("BertPreTokenizer"))) {
    return iree_tokenizer_parse_bert_pretokenizer(pretokenizer_json, allocator,
                                                  out_transform);
  }
  if (iree_string_view_equal(type_value, IREE_SV("Whitespace")) ||
      iree_string_view_equal(type_value, IREE_SV("WhitespaceSplit"))) {
    return iree_tokenizer_parse_whitespace_pretokenizer(
        pretokenizer_json, allocator, out_transform);
  }
  if (iree_string_view_equal(type_value, IREE_SV("ByteLevel"))) {
    return iree_tokenizer_parse_byte_level_pretokenizer(
        pretokenizer_json, allocator, out_transform);
  }
  if (iree_string_view_equal(type_value, IREE_SV("Metaspace"))) {
    return iree_tokenizer_parse_metaspace_pretokenizer(
        pretokenizer_json, allocator, out_transform);
  }
  if (iree_string_view_equal(type_value, IREE_SV("Sequence"))) {
    return iree_tokenizer_parse_sequence_pretokenizer(pretokenizer_json,
                                                      allocator, out_transform);
  }
  if (iree_string_view_equal(type_value, IREE_SV("Split"))) {
    return iree_tokenizer_parse_split_pretokenizer(pretokenizer_json, allocator,
                                                   out_transform);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported pre_tokenizer type '%.*s'",
                          (int)type_value.size, type_value.data);
}

iree_status_t iree_tokenizer_text_transform_parse_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  IREE_ASSERT_ARGUMENT(out_transform);
  iree_tokenizer_text_transform_initialize_none(out_transform);

  // Look up the pre_tokenizer field (optional).
  iree_string_view_t pretokenizer_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json_root, IREE_SV("pre_tokenizer"), &pretokenizer_value));

  if (pretokenizer_value.size > 0 &&
      !iree_string_view_equal(pretokenizer_value, IREE_SV("null"))) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_text_transform_parse_pretokenizer(
        pretokenizer_value, allocator, out_transform));
  }

  // Parse normalizer field (optional). Set on transform's embedded normalizer.
  iree_status_t status = iree_tokenizer_normalizer_parse_json(
      json_root, allocator, &out_transform->normalizer);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_text_transform_deinitialize(out_transform);
    return status;
  }

  return iree_ok_status();
}
