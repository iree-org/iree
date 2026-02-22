// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/segmenter_json.h"

#include "iree/base/internal/json.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/segmenter/bert.h"
#include "iree/tokenizer/segmenter/digits.h"
#include "iree/tokenizer/segmenter/metaspace.h"
#include "iree/tokenizer/segmenter/punctuation.h"
#include "iree/tokenizer/segmenter/sequence.h"
#include "iree/tokenizer/segmenter/split.h"
#include "iree/tokenizer/segmenter/whitespace.h"

//===----------------------------------------------------------------------===//
// ByteLevel Pre-Tokenizer
//===----------------------------------------------------------------------===//

// Maximum unescaped regex pattern length for Split pre_tokenizers.
// Real-world patterns (GPT-2, cl100k_base, Llama-3) are all < 256 bytes.
#define IREE_TOKENIZER_SPLIT_MAX_REGEX_LENGTH 1024

// GPT-2 regex pattern for ByteLevel pre_tokenizer with use_regex=true.
// This pattern matches tokens (contractions, words, numbers, punctuation,
// whitespace) and is used with Split in ISOLATED mode.
static const char kGPT2RegexPattern[] =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
    "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// Parses a ByteLevel pre_tokenizer and creates a Split segmenter.
// JSON structure:
//   {
//     "type": "ByteLevel",
//     "add_prefix_space": false,   // Reported via out_flags
//     "trim_offsets": true,        // Reported via out_flags
//     "use_regex": true            // If true (default), use GPT-2 regex
//   }
//
// ByteLevel pre_tokenizer does two things:
//   - Regex splitting to find word boundaries (handled by Split segmenter)
//   - Byte→Unicode encoding (handled by BPE model with byte_level_input flag)
//
// When use_regex=false, no segmenter is created (passthrough).
static iree_status_t iree_tokenizer_parse_byte_level_pre_tokenizer(
    iree_string_view_t pre_tokenizer_value, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter,
    iree_tokenizer_huggingface_pre_tokenizer_flags_t* out_flags) {
  *out_segmenter = NULL;

  // Report ByteLevel presence to the orchestrator.
  *out_flags |= IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL;

  // Parse add_prefix_space (required, no serde default in HF).
  bool add_prefix_space = false;
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(
      pre_tokenizer_value, IREE_SV("add_prefix_space"), &add_prefix_space));
  if (add_prefix_space) {
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_ADD_PREFIX_SPACE;
  }

  // Parse trim_offsets (required, no serde default in HF).
  // This flag is passed through to the postprocessor and only affects offset
  // tracking (training). Inference without offset tracking is unaffected.
  bool trim_offsets = false;
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(
      pre_tokenizer_value, IREE_SV("trim_offsets"), &trim_offsets));
  if (trim_offsets) {
    *out_flags |= IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_TRIM_OFFSETS;
  }

  // Parse use_regex (optional, default true).
  bool use_regex = true;
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_bool(pre_tokenizer_value, IREE_SV("use_regex"),
                                /*default_value=*/true, &use_regex));

  // If use_regex=false, no segmenter needed (passthrough).
  if (!use_regex) {
    return iree_ok_status();
  }

  // ByteLevel with use_regex=true produces word-level segments via the GPT-2
  // regex, enabling the word cache optimization in the BPE model.
  *out_flags |= IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT;

  // Compile the GPT-2 regex pattern to a DFA.
  iree_tokenizer_regex_dfa_t dfa;
  uint8_t* dfa_data = NULL;
  iree_tokenizer_regex_compile_error_t compile_error = {0};
  iree_status_t status = iree_tokenizer_regex_compile_and_load(
      iree_make_cstring_view(kGPT2RegexPattern),
      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, allocator, &dfa, &dfa_data,
      &compile_error);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(
        status, "failed to compile GPT-2 regex pattern at position %zu: %s",
        compile_error.position, compile_error.message);
  }

  // Create a Split segmenter with ISOLATED behavior (GPT-2 default).
  status = iree_tokenizer_segmenter_split_allocate(
      dfa, dfa_data, IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED,
      /*invert=*/false, allocator, out_segmenter);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, dfa_data);
    return status;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Split Pre-Tokenizer
//===----------------------------------------------------------------------===//

// Maps a HuggingFace SplitDelimiterBehavior string to our enum.
static iree_status_t iree_tokenizer_parse_split_behavior(
    iree_string_view_t behavior_str,
    iree_tokenizer_regex_split_behavior_t* out_behavior) {
  if (iree_string_view_equal(behavior_str, IREE_SV("Removed"))) {
    *out_behavior = IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED;
  } else if (iree_string_view_equal(behavior_str, IREE_SV("Isolated"))) {
    *out_behavior = IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED;
  } else if (iree_string_view_equal(behavior_str,
                                    IREE_SV("MergedWithPrevious"))) {
    *out_behavior = IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS;
  } else if (iree_string_view_equal(behavior_str, IREE_SV("MergedWithNext"))) {
    *out_behavior = IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT;
  } else if (iree_string_view_equal(behavior_str, IREE_SV("Contiguous"))) {
    *out_behavior = IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown Split behavior: '%.*s'",
                            (int)behavior_str.size, behavior_str.data);
  }
  return iree_ok_status();
}

// Parses a Split pre_tokenizer and creates a Split segmenter.
// JSON structure:
//   {
//     "type": "Split",
//     "pattern": {"Regex": "..."} or {"String": "..."},
//     "behavior": "Removed"|"Isolated"|"MergedWithPrevious"|...,
//     "invert": true|false
//   }
//
// All fields are required (matches HuggingFace behavior).
static iree_status_t iree_tokenizer_parse_split_pre_tokenizer(
    iree_string_view_t pre_tokenizer_value, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter) {
  *out_segmenter = NULL;

  // Get the pattern object (required) and extract the regex string.
  // Pattern is a tagged enum: {"Regex": "..."} or {"String": "..."}.
  iree_string_view_t pattern_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      pre_tokenizer_value, IREE_SV("pattern"), &pattern_value));

  // Check for Regex or String pattern. String patterns use optimized literal
  // matching; Regex patterns use the DFA engine.
  iree_string_view_t regex_str = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pattern_value, IREE_SV("Regex"), &regex_str));

  iree_string_view_t string_pattern = iree_string_view_empty();
  if (regex_str.size == 0) {
    IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
        pattern_value, IREE_SV("String"), &string_pattern));
    if (string_pattern.size == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Split pattern must contain 'Regex' or 'String'");
    }
  }

  // Parse behavior (required) and invert (required, no serde default in HF).
  iree_string_view_t behavior_str = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      pre_tokenizer_value, IREE_SV("behavior"), &behavior_str));
  iree_tokenizer_regex_split_behavior_t behavior;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_parse_split_behavior(behavior_str, &behavior));

  bool invert = false;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_bool(pre_tokenizer_value, IREE_SV("invert"), &invert));

  // Handle String patterns with optimized literal matching.
  if (string_pattern.size > 0) {
    // Unescape the JSON string to get the literal pattern.
    char pattern_buffer[IREE_TOKENIZER_SPLIT_MAX_REGEX_LENGTH];
    iree_host_size_t pattern_length = 0;
    IREE_RETURN_IF_ERROR(
        iree_json_unescape_string(string_pattern, sizeof(pattern_buffer),
                                  pattern_buffer, &pattern_length));
    iree_string_view_t pattern =
        iree_make_string_view(pattern_buffer, pattern_length);

    return iree_tokenizer_segmenter_split_literal_allocate(
        pattern, behavior, invert, allocator, out_segmenter);
  }

  // Handle Regex patterns with DFA engine.
  // Unescape the JSON string to get the actual regex pattern. JSON escape
  // sequences like \\p must be decoded to \p for the regex compiler.
  char regex_buffer[IREE_TOKENIZER_SPLIT_MAX_REGEX_LENGTH];
  iree_host_size_t regex_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(
      regex_str, sizeof(regex_buffer), regex_buffer, &regex_length));
  iree_string_view_t pattern =
      iree_make_string_view(regex_buffer, regex_length);

  // Compile the regex pattern to a DFA.
  iree_tokenizer_regex_dfa_t dfa;
  uint8_t* dfa_data = NULL;
  iree_tokenizer_regex_compile_error_t compile_error = {0};
  iree_status_t status = iree_tokenizer_regex_compile_and_load(
      pattern, IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, allocator, &dfa,
      &dfa_data, &compile_error);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(
        status, "failed to compile Split regex at position %" PRIhsz ": %s",
        compile_error.position, compile_error.message);
  }

  // Create the Split segmenter. On failure, free DFA data.
  status = iree_tokenizer_segmenter_split_allocate(
      dfa, dfa_data, behavior, invert, allocator, out_segmenter);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, dfa_data);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Punctuation Pre-Tokenizer
//===----------------------------------------------------------------------===//

// Parses a Punctuation pre_tokenizer and creates a Punctuation segmenter.
// JSON structure:
//   {
//     "type": "Punctuation",
//     "behavior": "Isolated"  // Optional, default "Isolated"
//   }
//
// The behavior field uses the same SplitDelimiterBehavior enum as Split:
// "Removed", "Isolated", "MergedWithPrevious", "MergedWithNext", "Contiguous".
static iree_status_t iree_tokenizer_parse_punctuation_pre_tokenizer(
    iree_string_view_t pre_tokenizer_value, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter) {
  *out_segmenter = NULL;

  // Parse behavior (optional, default "Isolated").
  iree_tokenizer_regex_split_behavior_t behavior =
      IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED;
  iree_string_view_t behavior_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pre_tokenizer_value, IREE_SV("behavior"), &behavior_value));
  if (behavior_value.size > 0) {
    IREE_RETURN_IF_ERROR(
        iree_tokenizer_parse_split_behavior(behavior_value, &behavior));
  }

  return iree_tokenizer_segmenter_punctuation_allocate(behavior, allocator,
                                                       out_segmenter);
}

//===----------------------------------------------------------------------===//
// Sequence Pre-Tokenizer
//===----------------------------------------------------------------------===//

// Parses a Sequence pre_tokenizer that chains multiple child pre-tokenizers.
// JSON structure:
//   {
//     "type": "Sequence",
//     "pretokenizers": [
//       {"type": "Split", ...},
//       {"type": "ByteLevel", ...},
//       ...
//     ]
//   }
//
// Children that parse to NULL (e.g., ByteLevel with use_regex=false) are
// skipped. If only one non-NULL child remains, it is returned directly
// without wrapping in a Sequence.
static iree_status_t iree_tokenizer_parse_sequence_pre_tokenizer(
    iree_string_view_t pre_tokenizer_value, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter,
    iree_tokenizer_huggingface_pre_tokenizer_flags_t* out_flags) {
  *out_segmenter = NULL;

  // Get the pretokenizers array (required).
  iree_string_view_t array_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      pre_tokenizer_value, IREE_SV("pretokenizers"), &array_value));

  iree_host_size_t array_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(array_value, &array_length));

  if (array_length == 0) {
    return iree_ok_status();
  }
  if (array_length > IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Sequence pretokenizers array too large (%" PRIhsz " > %d)",
        array_length, IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH);
  }

  // Parse each child recursively. Children may be NULL (e.g., ByteLevel with
  // use_regex=false returns passthrough).
  iree_tokenizer_segmenter_t*
      children[IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH];
  iree_host_size_t child_count = 0;
  iree_status_t status = iree_ok_status();

  for (iree_host_size_t i = 0; i < array_length && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t element_value = iree_string_view_empty();
    status = iree_json_array_get(array_value, i, &element_value);
    if (!iree_status_is_ok(status)) break;

    iree_tokenizer_segmenter_t* child = NULL;
    status = iree_tokenizer_huggingface_parse_segmenter(
        element_value, allocator, &child, out_flags);
    if (iree_status_is_ok(status) && child) {
      children[child_count++] = child;
    }
  }

  // If only one non-NULL child, return it directly (no sequence wrapper).
  if (iree_status_is_ok(status)) {
    if (child_count == 0) {
      return iree_ok_status();
    } else if (child_count == 1) {
      *out_segmenter = children[0];
      return iree_ok_status();
    }
    status = iree_tokenizer_segmenter_sequence_allocate(
        children, child_count, allocator, out_segmenter);
  }

  // On any failure (parsing or allocation), free already-parsed children.
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < child_count; ++i) {
      iree_tokenizer_segmenter_free(children[i]);
    }
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Metaspace Pre-Tokenizer
//===----------------------------------------------------------------------===//

// Parses a Metaspace pre_tokenizer and creates the segmenter.
// JSON structure:
//   {
//     "type": "Metaspace",
//     "replacement": "▁",        // Required (char, no serde default in HF)
//     "prepend_scheme": "first", // Optional, default "always"
//     "split": true              // Optional, default true
//   }
//
// The prepend_scheme field controls whether the replacement character is
// prepended to input before tokenization. "first" and "always" both require a
// prepend normalizer (signaled via out_flags); "never" skips prepending.
static iree_status_t iree_tokenizer_parse_metaspace_pre_tokenizer(
    iree_string_view_t pre_tokenizer_value, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter,
    iree_tokenizer_huggingface_pre_tokenizer_flags_t* out_flags) {
  *out_segmenter = NULL;
  // ALL Metaspace pre_tokenizers require space→replacement substitution.
  *out_flags |= IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_REPLACE;

  // Parse replacement character (required, no serde default in HF).
  uint32_t replacement_codepoint = IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT;
  iree_string_view_t replacement_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      pre_tokenizer_value, IREE_SV("replacement"), &replacement_value));
  IREE_RETURN_IF_ERROR(iree_json_parse_codepoint(
      replacement_value, IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT,
      &replacement_codepoint));

  // Parse prepend_scheme (optional, default "always").
  // "first" or "always": signal that a prepend normalizer is needed.
  // "never": no prepend. Any other value is invalid.
  //
  // Legacy: add_prefix_space=true is equivalent to prepend_scheme="always".
  // Some older tokenizer.json files use add_prefix_space instead of
  // prepend_scheme. We check add_prefix_space only if prepend_scheme is absent.
  iree_string_view_t prepend_scheme_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pre_tokenizer_value, IREE_SV("prepend_scheme"), &prepend_scheme_value));

  // Handle legacy add_prefix_space field if prepend_scheme is absent.
  bool use_legacy_add_prefix_space = false;
  if (prepend_scheme_value.size == 0) {
    bool add_prefix_space = false;
    IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(
        pre_tokenizer_value, IREE_SV("add_prefix_space"),
        /*default_value=*/false, &add_prefix_space));
    use_legacy_add_prefix_space = add_prefix_space;
  }

  if (use_legacy_add_prefix_space || prepend_scheme_value.size == 0 ||
      iree_string_view_equal(prepend_scheme_value, IREE_SV("null")) ||
      iree_string_view_equal(prepend_scheme_value, IREE_SV("always")) ||
      iree_string_view_equal(prepend_scheme_value, IREE_SV("first"))) {
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_PREPEND;
  } else if (!iree_string_view_equal(prepend_scheme_value, IREE_SV("never"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid Metaspace prepend_scheme: '%.*s'",
                            (int)prepend_scheme_value.size,
                            prepend_scheme_value.data);
  }

  // Parse split field (optional, default true).
  bool split_enabled = true;
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_bool(pre_tokenizer_value, IREE_SV("split"),
                                /*default_value=*/true, &split_enabled));

  // Metaspace with split=true produces word-level segments by splitting on
  // spaces, enabling the word cache optimization in the BPE model.
  if (split_enabled) {
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT;
  }

  return iree_tokenizer_segmenter_metaspace_allocate(
      replacement_codepoint, split_enabled, allocator, out_segmenter);
}

//===----------------------------------------------------------------------===//
// Top-Level Segmenter Dispatcher
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_huggingface_parse_segmenter(
    iree_string_view_t pre_tokenizer_value, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter,
    iree_tokenizer_huggingface_pre_tokenizer_flags_t* out_flags) {
  IREE_ASSERT_ARGUMENT(out_segmenter);
  IREE_ASSERT_ARGUMENT(out_flags);
  *out_segmenter = NULL;

  // Handle null pre_tokenizer.
  if (iree_string_view_equal(pre_tokenizer_value, IREE_SV("null"))) {
    return iree_ok_status();
  }

  // Get pre_tokenizer type.
  iree_string_view_t type_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      pre_tokenizer_value, IREE_SV("type"), &type_value));

  // Dispatch based on type.
  if (iree_string_view_equal(type_value, IREE_SV("BertPreTokenizer"))) {
    // BertPreTokenizer produces word-level segments (whitespace + punctuation
    // splitting). No additional JSON fields to parse.
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT;
    return iree_tokenizer_segmenter_bert_allocate(allocator, out_segmenter);
  } else if (iree_string_view_equal(type_value, IREE_SV("ByteLevel"))) {
    return iree_tokenizer_parse_byte_level_pre_tokenizer(
        pre_tokenizer_value, allocator, out_segmenter, out_flags);
  } else if (iree_string_view_equal(type_value, IREE_SV("Digits"))) {
    // Digits pre-tokenizer splits on digit boundaries. Produces word-level
    // segments separating digits from non-digits.
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT;
    bool individual_digits = false;
    IREE_RETURN_IF_ERROR(iree_json_try_lookup_bool(
        pre_tokenizer_value, IREE_SV("individual_digits"),
        /*default_value=*/false, &individual_digits));
    return iree_tokenizer_segmenter_digits_allocate(individual_digits,
                                                    allocator, out_segmenter);
  } else if (iree_string_view_equal(type_value, IREE_SV("Metaspace"))) {
    return iree_tokenizer_parse_metaspace_pre_tokenizer(
        pre_tokenizer_value, allocator, out_segmenter, out_flags);
  } else if (iree_string_view_equal(type_value, IREE_SV("Punctuation"))) {
    // Punctuation pre-tokenizer produces word-level segments by isolating
    // punctuation characters.
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT;
    return iree_tokenizer_parse_punctuation_pre_tokenizer(
        pre_tokenizer_value, allocator, out_segmenter);
  } else if (iree_string_view_equal(type_value, IREE_SV("Split"))) {
    // Split pre-tokenizer always produces word-level segments via regex.
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT;
    return iree_tokenizer_parse_split_pre_tokenizer(pre_tokenizer_value,
                                                    allocator, out_segmenter);
  } else if (iree_string_view_equal(type_value, IREE_SV("Sequence"))) {
    return iree_tokenizer_parse_sequence_pre_tokenizer(
        pre_tokenizer_value, allocator, out_segmenter, out_flags);
  } else if (iree_string_view_equal(type_value, IREE_SV("Whitespace")) ||
             iree_string_view_equal(type_value, IREE_SV("WhitespaceSplit"))) {
    // Whitespace and WhitespaceSplit are equivalent: split on whitespace,
    // discard whitespace, keep non-whitespace segments. No additional JSON
    // fields to parse.
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT;
    *out_flags |=
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_HAS_WHITESPACE_SPLIT;
    return iree_tokenizer_segmenter_whitespace_allocate(allocator,
                                                        out_segmenter);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "pre_tokenizer type not yet supported: '%.*s'",
                          (int)type_value.size, type_value.data);
}
