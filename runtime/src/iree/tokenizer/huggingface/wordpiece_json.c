// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/wordpiece_json.h"

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "iree/tokenizer/huggingface/added_tokens.h"
#include "iree/tokenizer/huggingface/decoder_json.h"
#include "iree/tokenizer/huggingface/literals_json.h"
#include "iree/tokenizer/huggingface/postprocessor_json.h"
#include "iree/tokenizer/huggingface/transform_json.h"
#include "iree/tokenizer/huggingface/vocab_json.h"
#include "iree/tokenizer/vocab_builder.h"
#include "iree/tokenizer/wordpiece.h"

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_import_wordpiece_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_vocab_t** out_vocab) {
  IREE_ASSERT_ARGUMENT(out_vocab);
  *out_vocab = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Extract and validate the model object using shared utilities.
  iree_string_view_t model;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_json_extract_model(json, &model));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_json_validate_model_type(model, IREE_SV("WordPiece")));

  // Validate model object keys.
  static const iree_string_view_t kModelAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("vocab"),
      IREE_SVL("unk_token"),
      IREE_SVL("continuing_subword_prefix"),
      IREE_SVL("max_input_chars_per_word"),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_validate_object_keys(model, kModelAllowedKeys,
                                         IREE_ARRAYSIZE(kModelAllowedKeys)));

  // Phase 1: Parse added_tokens into temporary storage.
  iree_tokenizer_added_tokens_t added_tokens;
  iree_tokenizer_added_tokens_initialize(&added_tokens, allocator);
  iree_status_t status =
      iree_tokenizer_added_tokens_parse_json(json, allocator, &added_tokens);

  // Estimate vocab capacity from JSON size to avoid a separate counting pass.
  // The vocab_builder grows dynamically if the estimate is too low.
  iree_host_size_t estimated_vocab_count =
      iree_tokenizer_json_estimate_vocab_capacity(json.size);
  iree_host_size_t capacity = estimated_vocab_count + added_tokens.count;

  // Allocate builder.
  iree_tokenizer_vocab_builder_t* builder = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_tokenizer_vocab_builder_allocate(capacity, allocator, &builder);
  }

  // Phase 3: Parse vocab entries (checking against added_tokens for attrs).
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_json_parse_vocab(model, allocator, &added_tokens,
                                             builder);
  }

  // Phase 4: Add missing tokens and set special IDs from added_tokens,
  // OR use fallback detection if added_tokens was empty.
  if (iree_status_is_ok(status)) {
    if (added_tokens.count > 0) {
      status = iree_tokenizer_added_tokens_finalize(&added_tokens, builder);
    } else {
      status = iree_tokenizer_detect_specials_from_vocab(model, builder);
    }
  }

  // Build final vocab on success, cleanup on failure.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(builder, out_vocab);
  } else {
    if (builder) iree_tokenizer_vocab_builder_free(builder);
  }
  iree_tokenizer_added_tokens_deinitialize(&added_tokens);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// WordPiece Config Parsing
//===----------------------------------------------------------------------===//

// Parses WordPiece config from JSON. On success, |out_prefix_storage| receives
// an allocated copy of the prefix (caller must free), or NULL if default.
static iree_status_t iree_tokenizer_wordpiece_parse_json_config(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_wordpiece_config_t* out_config, char** out_prefix_storage) {
  *out_config = IREE_TOKENIZER_WORDPIECE_CONFIG_DEFAULT;
  *out_prefix_storage = NULL;

  iree_string_view_t model;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("model"), &model));

  // Extract max_input_chars_per_word if present (no allocation needed).
  iree_string_view_t max_chars_json;
  iree_status_t status = iree_json_try_lookup_object_value(
      model, IREE_SV("max_input_chars_per_word"), &max_chars_json);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  if (max_chars_json.size > 0) {
    int64_t max_chars = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_int64(max_chars_json, &max_chars));
    if (max_chars <= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "max_input_chars_per_word must be positive");
    }
    // Compare as unsigned after validating positive to avoid signed overflow.
    if ((uint64_t)max_chars > (uint64_t)IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "max_input_chars_per_word exceeds host size max");
    }
    out_config->max_input_chars_per_word = (iree_host_size_t)max_chars;
  }

  // Extract continuing_subword_prefix if present (allocates storage).
  // An empty string ("") is a valid value that disables subword prefixing.
  iree_string_view_t prefix_json;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      model, IREE_SV("continuing_subword_prefix"), &prefix_json));
  if (prefix_json.size > 0 &&
      !iree_string_view_equal(prefix_json, IREE_SV("null"))) {
    iree_host_size_t prefix_length;
    IREE_RETURN_IF_ERROR(
        iree_json_unescape_string(prefix_json, 0, NULL, &prefix_length));
    if (prefix_length > 0) {
      char* prefix_storage = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, prefix_length,
                                                 (void**)&prefix_storage));
      status = iree_json_unescape_string(prefix_json, prefix_length,
                                         prefix_storage, &prefix_length);
      if (!iree_status_is_ok(status)) {
        iree_allocator_free(allocator, prefix_storage);
        return status;
      }
      out_config->continuing_subword_prefix =
          iree_make_string_view(prefix_storage, prefix_length);
      *out_prefix_storage = prefix_storage;
    } else {
      // Empty string: use empty prefix (no subword marking).
      // Point to a static empty string so .data is non-NULL.
      static const char empty_string[] = "";
      out_config->continuing_subword_prefix =
          iree_make_string_view(empty_string, 0);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// WordPiece Tokenizer Factory
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_from_wordpiece_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Initialize transform, decoder, and postprocessor to safe defaults.
  // This ensures cleanup is safe if vocab import or any other step fails.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_none(&transform);
  iree_tokenizer_decoder_t decoder;
  iree_tokenizer_decoder_initialize_none(&decoder);
  iree_tokenizer_postprocessor_t postprocessor;
  iree_tokenizer_postprocessor_initialize_none(&postprocessor);

  // Import vocab from JSON.
  iree_tokenizer_vocab_t* vocab = NULL;
  iree_status_t status =
      iree_tokenizer_vocab_import_wordpiece_json(json, allocator, &vocab);

  // Parse pre-tokenizer transform from JSON (includes normalizer).
  if (iree_status_is_ok(status)) {
    status =
        iree_tokenizer_text_transform_parse_json(json, allocator, &transform);
  }

  // Parse decoder from JSON.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_decoder_parse_json(json, &decoder);
  }

  // Parse post_processor from JSON (controls special token insertion).
  // BERT/WordPiece models typically have: [CLS] $A [SEP] or [CLS] $A [SEP] $B
  // [SEP]
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_postprocessor_parse_json(json, allocator,
                                                     &postprocessor);
  }

  // Parse config from JSON.
  iree_tokenizer_wordpiece_config_t config;
  char* prefix_storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_wordpiece_parse_json_config(
        json, allocator, &config, &prefix_storage);
  }

  // Create tokenizer with config. Allocate consumes vocab and prefix_storage
  // unconditionally (frees on failure, owns on success).
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_wordpiece_allocate(vocab, &config, prefix_storage,
                                               allocator, &tokenizer);
  } else {
    if (prefix_storage) iree_allocator_free(allocator, prefix_storage);
    iree_tokenizer_vocab_free(vocab);
  }
  vocab = NULL;           // Consumed by allocate or freed above.
  prefix_storage = NULL;  // Consumed by allocate or freed above.

  // Parse added_tokens (literals) from JSON.
  // Literals are string patterns that map directly to token IDs, bypassing
  // the pre-tokenizer. Tokens with lstrip/rstrip/single_word/normalized flags
  // are intercepted before the transform to prevent incorrect splitting.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_literals_parse_json(json, &tokenizer->literals);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_literals_finalize(&tokenizer->literals);
  }

  // On success, set transform/decoder/postprocessor.
  if (iree_status_is_ok(status)) {
    tokenizer->transform = transform;
    tokenizer->decoder = decoder;
    tokenizer->postprocessor = postprocessor;
    *out_tokenizer = tokenizer;
  } else {
    // Cleanup on failure (vocab and prefix_storage already freed above).
    iree_tokenizer_postprocessor_deinitialize(&postprocessor);
    iree_tokenizer_decoder_deinitialize(&decoder);
    iree_tokenizer_text_transform_deinitialize(&transform);
    if (tokenizer) {
      iree_tokenizer_free(tokenizer);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
