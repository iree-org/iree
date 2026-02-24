// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

#include "iree/base/internal/json.h"
#include "iree/tokenizer/format/huggingface/added_tokens_json.h"
#include "iree/tokenizer/format/huggingface/decoder_json.h"
#include "iree/tokenizer/format/huggingface/model_json.h"
#include "iree/tokenizer/format/huggingface/normalizer_json.h"
#include "iree/tokenizer/format/huggingface/postprocessor_json.h"
#include "iree/tokenizer/format/huggingface/segmenter_json.h"
#include "iree/tokenizer/normalizer/prepend.h"
#include "iree/tokenizer/normalizer/regex_replace.h"
#include "iree/tokenizer/normalizer/replace.h"
#include "iree/tokenizer/normalizer/sequence.h"
#include "iree/tokenizer/normalizer/strip.h"
#include "iree/tokenizer/special_tokens.h"

//===----------------------------------------------------------------------===//
// Allowed Top-Level Keys
//===----------------------------------------------------------------------===//

// Top-level keys in tokenizer.json.
// Reference: tokenizers/src/tokenizer/serialization.rs
//
// The HuggingFace tokenizers library accepts any keys (unknown keys are
// silently ignored), but we require strict validation to catch typos and
// ensure forward compatibility. We explicitly list all known keys.
static const iree_string_view_t kTopLevelAllowedKeys[] = {
    IREE_SVL("version"),         // Optional: "1.0"
    IREE_SVL("truncation"),      // Optional: truncation params
    IREE_SVL("padding"),         // Optional: padding params
    IREE_SVL("added_tokens"),    // Optional: array of added tokens
    IREE_SVL("normalizer"),      // Optional: normalizer config
    IREE_SVL("pre_tokenizer"),   // Optional: pre-tokenizer config
    IREE_SVL("post_processor"),  // Optional: post-processor config
    IREE_SVL("decoder"),         // Optional: decoder config
    IREE_SVL("model"),           // Required: model config
};

//===----------------------------------------------------------------------===//
// Special Tokens Builder
//===----------------------------------------------------------------------===//

// Builds special_tokens collections from added_tokens.
//
// ALL added tokens with `normalized=true` are included (not just those with
// `special=true`) because they must be matched BEFORE the segmenter runs.
// Without this, ByteLevel segmentation transforms the input before these
// tokens can match, breaking tokenizers like GPT-NeoX that have multi-space
// tokens (e.g., "  " at ID 50276).
//
// Tokens are split by their normalized flag:
//   - out_special_tokens: normalized=false, matched in raw input before
//     normalization runs (e.g., <|endoftext|>)
//   - out_special_tokens_post_norm: normalized=true, matched after
//     normalization transforms the input but BEFORE segmentation
static iree_status_t iree_tokenizer_huggingface_build_special_tokens(
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_allocator_t allocator,
    iree_tokenizer_special_tokens_t* out_special_tokens,
    iree_tokenizer_special_tokens_t* out_special_tokens_post_norm) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Initialize outputs to empty state.
  iree_tokenizer_special_tokens_initialize(out_special_tokens);
  iree_tokenizer_special_tokens_initialize(out_special_tokens_post_norm);

  // Count tokens to process: special tokens OR tokens with normalized=true.
  // Non-special tokens with normalized=true must be matched before
  // segmentation.
  iree_host_size_t match_count = 0;
  for (iree_host_size_t i = 0; i < added_tokens->count; ++i) {
    const iree_tokenizer_huggingface_added_token_t* token =
        iree_tokenizer_huggingface_added_tokens_get(added_tokens, i);
    bool is_special = iree_any_bit_set(
        token->flags, IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL);
    bool is_normalized = iree_any_bit_set(
        token->flags, IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_NORMALIZED);
    // Include if special OR if normalized (needs pre-segmentation matching).
    if (is_special || is_normalized) {
      ++match_count;
    }
  }

  // Early exit if nothing to process.
  if (match_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, match_count);

  // Build two separate collections: pre-normalization and post-normalization.
  iree_tokenizer_special_tokens_builder_t builder_pre_norm;
  iree_tokenizer_special_tokens_builder_t builder_post_norm;
  iree_tokenizer_special_tokens_builder_initialize(allocator,
                                                   &builder_pre_norm);
  iree_tokenizer_special_tokens_builder_initialize(allocator,
                                                   &builder_post_norm);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < added_tokens->count && iree_status_is_ok(status); ++i) {
    const iree_tokenizer_huggingface_added_token_t* token =
        iree_tokenizer_huggingface_added_tokens_get(added_tokens, i);
    bool is_special = iree_any_bit_set(
        token->flags, IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL);
    bool is_normalized = iree_any_bit_set(
        token->flags, IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_NORMALIZED);

    // Process if special OR if normalized (needs pre-segmentation matching).
    if (is_special || is_normalized) {
      iree_string_view_t content =
          iree_tokenizer_huggingface_added_token_content(added_tokens, token);

      // Convert HuggingFace flags to special_tokens flags.
      iree_tokenizer_special_token_flags_t special_flags =
          IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE;
      if (iree_any_bit_set(
              token->flags,
              IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_LSTRIP)) {
        special_flags |= IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP;
      }
      if (iree_any_bit_set(
              token->flags,
              IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_RSTRIP)) {
        special_flags |= IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP;
      }
      if (iree_any_bit_set(
              token->flags,
              IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SINGLE_WORD)) {
        special_flags |= IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_SINGLE_WORD;
      }

      // Route to appropriate builder based on normalized flag.
      if (is_normalized) {
        // normalized=true: match after normalization but before segmentation.
        status = iree_tokenizer_special_tokens_builder_add(
            &builder_post_norm, content, token->id, special_flags);
      } else {
        // normalized=false (default): match before normalization.
        status = iree_tokenizer_special_tokens_builder_add(
            &builder_pre_norm, content, token->id, special_flags);
      }
    }
  }

  // Build both collections (builders are cleaned up after).
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_special_tokens_builder_build(
        &builder_pre_norm, allocator, out_special_tokens);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_special_tokens_builder_build(
        &builder_post_norm, allocator, out_special_tokens_post_norm);
  }

  iree_tokenizer_special_tokens_builder_deinitialize(&builder_pre_norm);
  iree_tokenizer_special_tokens_builder_deinitialize(&builder_post_norm);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_parse_huggingface_json(
    iree_string_view_t json, iree_tokenizer_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // Validate inputs.
  if (json.size == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty JSON input");
  }

  if (iree_status_is_ok(status)) {
    status = iree_json_validate_object_keys(
        json, kTopLevelAllowedKeys, IREE_ARRAYSIZE(kTopLevelAllowedKeys));
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate(
          status, IREE_SV("in tokenizer.json top-level object"));
    }
  }

  // Validate version if present.
  iree_string_view_t version_value = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = iree_json_try_lookup_object_value(json, IREE_SV("version"),
                                               &version_value);
  }
  if (iree_status_is_ok(status) && version_value.size > 0 &&
      !iree_string_view_equal(version_value, IREE_SV("1.0"))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported tokenizer.json version: '%.*s' "
                              "(only '1.0' is supported)",
                              (int)version_value.size, version_value.data);
  }

  // Parse added_tokens first as other section parsers may need to extract
  // information from it during construction.
  iree_allocator_t allocator = builder->allocator;
  iree_tokenizer_huggingface_added_tokens_t added_tokens = {0};
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_huggingface_parse_added_tokens_json(json, allocator,
                                                                &added_tokens);
  }

  // Parse normalizer section (optional).
  iree_tokenizer_normalizer_t* normalizer = NULL;
  if (iree_status_is_ok(status)) {
    iree_string_view_t normalizer_value = iree_string_view_empty();
    status = iree_json_try_lookup_object_value(json, IREE_SV("normalizer"),
                                               &normalizer_value);
    if (iree_status_is_ok(status) && normalizer_value.size > 0 &&
        !iree_string_view_equal(normalizer_value, IREE_SV("null"))) {
      status = iree_tokenizer_huggingface_parse_normalizer(
          normalizer_value, allocator, &normalizer);
    }
  }

  // Parse pre_tokenizer section → segmenter + format-level flags (optional).
  iree_tokenizer_segmenter_t* segmenter = NULL;
  iree_tokenizer_huggingface_pre_tokenizer_flags_t pre_tokenizer_flags =
      IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE;
  iree_string_view_t pre_tokenizer_value = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = iree_json_try_lookup_object_value(json, IREE_SV("pre_tokenizer"),
                                               &pre_tokenizer_value);
    if (iree_status_is_ok(status) && pre_tokenizer_value.size > 0 &&
        !iree_string_view_equal(pre_tokenizer_value, IREE_SV("null"))) {
      status = iree_tokenizer_huggingface_parse_segmenter(
          pre_tokenizer_value, allocator, &segmenter, &pre_tokenizer_flags);
    }
  }
  if (iree_status_is_ok(status) && segmenter) {
    iree_tokenizer_builder_set_segmenter(builder, segmenter);
  }

  // Synthesize normalizers from Metaspace pre_tokenizer flags.
  // Chain order: [existing_normalizer, strip_right, replace, prepend].
  // U+2581 LOWER ONE EIGHTH BLOCK = 0xE2 0x96 0x81 in UTF-8.
  static const char kMetaspaceReplacement[] = "\xE2\x96\x81";

  // When WhitespaceSplit and Metaspace are both present, strip trailing
  // whitespace BEFORE the space→▁ replacement. This matches HuggingFace's
  // behavior where WhitespaceSplit discards trailing whitespace before
  // Metaspace runs. Without this, trailing spaces become trailing ▁ segments
  // that produce spurious tokens.
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(
          pre_tokenizer_flags,
          IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_HAS_WHITESPACE_SPLIT |
              IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_REPLACE)) {
    iree_tokenizer_normalizer_t* strip_normalizer = NULL;
    status = iree_tokenizer_normalizer_strip_allocate(
        /*strip_left=*/false, /*strip_right=*/true, allocator,
        &strip_normalizer);
    if (iree_status_is_ok(status)) {
      if (normalizer) {
        iree_tokenizer_normalizer_t* children[2] = {normalizer,
                                                    strip_normalizer};
        iree_tokenizer_normalizer_t* sequence = NULL;
        status = iree_tokenizer_normalizer_sequence_allocate(
            children, 2, allocator, &sequence);
        if (iree_status_is_ok(status)) {
          normalizer = sequence;
        } else {
          iree_tokenizer_normalizer_free(strip_normalizer);
        }
      } else {
        normalizer = strip_normalizer;
      }
    }
  }

  // Synthesize a replace normalizer for space→▁ substitution. The Metaspace
  // pre_tokenizer requires all 0x20 bytes to be replaced with the replacement
  // character so that the vocabulary can distinguish word boundaries.
  //
  // When WhitespaceSplit is also present, consecutive whitespace must collapse
  // to a single ▁. HuggingFace's Sequence[WhitespaceSplit, Metaspace] works by:
  //   1. WhitespaceSplit removes whitespace, producing ["word1", "word2"]
  //   2. Metaspace adds ▁ prefix per segment: ["▁word1", "▁word2"]
  // Our architecture runs normalizers before segmenters, so we collapse during
  // replacement: "  multiple   spaces" → "▁multiple▁spaces" (not
  // "▁▁multiple▁▁▁spaces").
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(
          pre_tokenizer_flags,
          IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_REPLACE)) {
    iree_tokenizer_normalizer_t* replace_normalizer = NULL;
    if (iree_any_bit_set(
            pre_tokenizer_flags,
            IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_HAS_WHITESPACE_SPLIT)) {
      // Collapse consecutive whitespace to single ▁ (regex: "\s+" → "▁").
      // WhitespaceSplit treats all Unicode whitespace as word boundaries, so
      // spaces, tabs, and newlines all become ▁ prefix on the following word.
      status = iree_tokenizer_normalizer_regex_replace_allocate(
          IREE_SV("\\s+"), iree_make_string_view(kMetaspaceReplacement, 3),
          allocator, &replace_normalizer);
    } else {
      // Simple 1:1 replacement (each space becomes ▁).
      status = iree_tokenizer_normalizer_replace_allocate(
          IREE_SV(" "), iree_make_string_view(kMetaspaceReplacement, 3),
          allocator, &replace_normalizer);
    }
    if (iree_status_is_ok(status)) {
      if (normalizer) {
        iree_tokenizer_normalizer_t* children[2] = {normalizer,
                                                    replace_normalizer};
        iree_tokenizer_normalizer_t* sequence = NULL;
        status = iree_tokenizer_normalizer_sequence_allocate(
            children, 2, allocator, &sequence);
        if (iree_status_is_ok(status)) {
          normalizer = sequence;
        } else {
          iree_tokenizer_normalizer_free(replace_normalizer);
        }
      } else {
        normalizer = replace_normalizer;
      }
    }
  }

  // Synthesize a prepend normalizer when the Metaspace pre_tokenizer has
  // prepend_scheme="first" or "always". The prepend emits the replacement
  // character (▁) before the first byte of input, which is required for
  // SentencePiece-style vocabularies where word-initial tokens include the
  // metaspace marker.
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(
          pre_tokenizer_flags,
          IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_PREPEND)) {
    iree_tokenizer_normalizer_t* prepend_normalizer = NULL;
    status = iree_tokenizer_normalizer_prepend_allocate(
        iree_make_string_view(kMetaspaceReplacement, 3),
        /*skip_if_prefix_matches=*/true, allocator, &prepend_normalizer);
    if (iree_status_is_ok(status)) {
      if (normalizer) {
        iree_tokenizer_normalizer_t* children[2] = {normalizer,
                                                    prepend_normalizer};
        iree_tokenizer_normalizer_t* sequence = NULL;
        status = iree_tokenizer_normalizer_sequence_allocate(
            children, 2, allocator, &sequence);
        if (iree_status_is_ok(status)) {
          normalizer = sequence;
        } else {
          iree_tokenizer_normalizer_free(prepend_normalizer);
        }
      } else {
        normalizer = prepend_normalizer;
      }
    }
  }

  // Synthesize a prepend normalizer when the ByteLevel pre_tokenizer has
  // add_prefix_space=true. The prepend emits a space before the first byte of
  // input, ensuring the first token gets a leading space marker in GPT-2/Llama
  // style vocabularies.
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(
          pre_tokenizer_flags,
          IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_ADD_PREFIX_SPACE)) {
    iree_tokenizer_normalizer_t* prepend_normalizer = NULL;
    status = iree_tokenizer_normalizer_prepend_allocate(
        IREE_SV(" "), /*skip_if_prefix_matches=*/true, allocator,
        &prepend_normalizer);
    if (iree_status_is_ok(status)) {
      if (normalizer) {
        iree_tokenizer_normalizer_t* children[2] = {normalizer,
                                                    prepend_normalizer};
        iree_tokenizer_normalizer_t* sequence = NULL;
        status = iree_tokenizer_normalizer_sequence_allocate(
            children, 2, allocator, &sequence);
        if (iree_status_is_ok(status)) {
          normalizer = sequence;
        } else {
          iree_tokenizer_normalizer_free(prepend_normalizer);
        }
      } else {
        normalizer = prepend_normalizer;
      }
    }
  }

  if (iree_status_is_ok(status) && normalizer) {
    iree_tokenizer_builder_set_normalizer(builder, normalizer);
  } else if (normalizer) {
    iree_tokenizer_normalizer_free(normalizer);
    normalizer = NULL;
  }

  // Parse decoder section (optional).
  iree_tokenizer_decoder_t* decoder = NULL;
  if (iree_status_is_ok(status)) {
    iree_string_view_t decoder_value = iree_string_view_empty();
    status = iree_json_try_lookup_object_value(json, IREE_SV("decoder"),
                                               &decoder_value);
    if (iree_status_is_ok(status) && decoder_value.size > 0 &&
        !iree_string_view_equal(decoder_value, IREE_SV("null"))) {
      status = iree_tokenizer_huggingface_parse_decoder(decoder_value,
                                                        allocator, &decoder);
    }
  }
  if (iree_status_is_ok(status) && decoder) {
    iree_tokenizer_builder_set_decoder(builder, decoder);
  }

  // Parse post_processor section (optional).
  if (iree_status_is_ok(status)) {
    iree_string_view_t postprocessor_value = iree_string_view_empty();
    status = iree_json_try_lookup_object_value(json, IREE_SV("post_processor"),
                                               &postprocessor_value);
    if (iree_status_is_ok(status) && postprocessor_value.size > 0 &&
        !iree_string_view_equal(postprocessor_value, IREE_SV("null"))) {
      iree_tokenizer_postprocessor_t postprocessor = {0};
      status = iree_tokenizer_huggingface_parse_postprocessor(
          postprocessor_value, &postprocessor);
      if (iree_status_is_ok(status)) {
        iree_tokenizer_builder_set_postprocessor(builder, postprocessor);
      }
    }
  }

  // Parse model section (returns both model and vocab).
  iree_tokenizer_model_t* parsed_model = NULL;
  iree_tokenizer_vocab_t* parsed_vocab = NULL;
  if (iree_status_is_ok(status)) {
    iree_string_view_t model_json = iree_string_view_empty();
    status = iree_json_lookup_object_value(json, IREE_SV("model"), &model_json);
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_huggingface_parse_model(
          model_json, &added_tokens, pre_tokenizer_flags, allocator,
          &parsed_model, &parsed_vocab);
    }
  }
  if (iree_status_is_ok(status) && parsed_model) {
    iree_tokenizer_builder_set_model(builder, parsed_model);
  }
  if (iree_status_is_ok(status) && parsed_vocab) {
    iree_tokenizer_builder_set_vocab(builder, parsed_vocab);
  }

  // Build special_tokens collections from added_tokens.
  // Includes all tokens with special=true, plus all tokens with normalized=true
  // (which must be matched before segmentation to handle multi-space tokens).
  // Tokens are routed by normalized flag: pre-norm (normalized=false) are
  // matched before the normalizer, post-norm (normalized=true) are matched
  // after normalization but before segmentation.
  if (iree_status_is_ok(status) && added_tokens.count > 0) {
    iree_tokenizer_special_tokens_t special_tokens;
    iree_tokenizer_special_tokens_t special_tokens_post_norm;
    status = iree_tokenizer_huggingface_build_special_tokens(
        &added_tokens, allocator, &special_tokens, &special_tokens_post_norm);
    if (iree_status_is_ok(status) &&
        !iree_tokenizer_special_tokens_is_empty(&special_tokens)) {
      iree_tokenizer_builder_set_special_tokens(builder, &special_tokens);
    }
    if (iree_status_is_ok(status) &&
        !iree_tokenizer_special_tokens_is_empty(&special_tokens_post_norm)) {
      iree_tokenizer_builder_set_special_tokens_post_norm(
          builder, &special_tokens_post_norm);
    }
  }

  // Free added_tokens now that all section parsers have extracted what they
  // need. Each component stores its own copy of relevant data.
  iree_tokenizer_huggingface_added_tokens_free(&added_tokens);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_from_huggingface_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Initialize builder (must be deinitialized on all paths).
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator, &builder);

  // Parse JSON into builder, then build tokenizer.
  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_builder_build(&builder, out_tokenizer);
  }

  iree_tokenizer_builder_deinitialize(&builder);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
