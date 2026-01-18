// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/unigram_json.h"

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "iree/tokenizer/huggingface/added_tokens.h"
#include "iree/tokenizer/huggingface/decoder_json.h"
#include "iree/tokenizer/huggingface/literals_json.h"
#include "iree/tokenizer/huggingface/postprocessor_json.h"
#include "iree/tokenizer/huggingface/transform_json.h"
#include "iree/tokenizer/huggingface/vocab_json.h"
#include "iree/tokenizer/unigram.h"
#include "iree/tokenizer/vocab_builder.h"

//===----------------------------------------------------------------------===//
// Parse Unigram Vocab Array
//===----------------------------------------------------------------------===//

// Context for parsing Unigram vocab entries.
typedef struct {
  iree_tokenizer_vocab_builder_t* builder;
  float* scores;
  iree_host_size_t score_capacity;
  iree_host_size_t current_index;
  iree_allocator_t allocator;
  char* unescape_buffer;
  iree_host_size_t unescape_capacity;
  iree_status_t status;
} iree_tokenizer_parse_unigram_ctx_t;

// Context for parsing a single vocab entry: ["token", score].
typedef struct {
  iree_tokenizer_parse_unigram_ctx_t* parent_ctx;
  iree_host_size_t element_count;
  iree_string_view_t token_text;
  float score;
} iree_tokenizer_unigram_entry_ctx_t;

// Visitor for vocab entry elements.
static iree_status_t iree_tokenizer_unigram_entry_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t value) {
  iree_tokenizer_unigram_entry_ctx_t* ctx =
      (iree_tokenizer_unigram_entry_ctx_t*)user_data;
  iree_tokenizer_parse_unigram_ctx_t* parent = ctx->parent_ctx;
  ctx->element_count++;

  if (index == 0) {
    // First element is the token string (in quotes).
    // Unescape the token.
    iree_host_size_t unescaped_length;
    iree_status_t status =
        iree_json_unescape_string(value, 0, NULL, &unescaped_length);
    if (!iree_status_is_ok(status)) {
      parent->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }

    // Ensure buffer capacity.
    if (unescaped_length > parent->unescape_capacity) {
      iree_host_size_t new_capacity = parent->unescape_capacity * 2;
      if (new_capacity < unescaped_length) new_capacity = unescaped_length;
      if (new_capacity < 256) new_capacity = 256;
      status = iree_allocator_realloc(parent->allocator, new_capacity,
                                      (void**)&parent->unescape_buffer);
      if (!iree_status_is_ok(status)) {
        parent->status = status;
        return iree_status_from_code(IREE_STATUS_CANCELLED);
      }
      parent->unescape_capacity = new_capacity;
    }

    status =
        iree_json_unescape_string(value, parent->unescape_capacity,
                                  parent->unescape_buffer, &unescaped_length);
    if (!iree_status_is_ok(status)) {
      parent->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }

    ctx->token_text =
        iree_make_string_view(parent->unescape_buffer, unescaped_length);
  } else if (index == 1) {
    // Second element is the score (float).
    double score_double = 0.0;
    iree_status_t status = iree_json_parse_double(value, &score_double);
    if (!iree_status_is_ok(status)) {
      parent->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
    ctx->score = (float)score_double;
  } else {
    parent->status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Unigram vocab entry must have exactly 2 elements, got more");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  return iree_ok_status();
}

// Visitor for top-level vocab array entries.
static iree_status_t iree_tokenizer_unigram_vocab_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t value) {
  iree_tokenizer_parse_unigram_ctx_t* ctx =
      (iree_tokenizer_parse_unigram_ctx_t*)user_data;

  // Parse the entry array: ["token", score].
  iree_tokenizer_unigram_entry_ctx_t entry_ctx = {
      .parent_ctx = ctx,
      .element_count = 0,
      .token_text = iree_string_view_empty(),
      .score = 0.0f,
  };

  iree_status_t status = iree_json_enumerate_array(
      value, iree_tokenizer_unigram_entry_visitor, &entry_ctx);

  // Check for errors during enumeration.
  if (!iree_status_is_ok(ctx->status)) {
    iree_status_ignore(status);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Verify we got exactly 2 elements.
  if (entry_ctx.element_count != 2) {
    ctx->status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                   "Unigram vocab entry must have exactly 2 "
                                   "elements, got %" PRIhsz,
                                   entry_ctx.element_count);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Add token to builder (with no special attributes).
  status = iree_tokenizer_vocab_builder_add_token(
      ctx->builder, entry_ctx.token_text, entry_ctx.score,
      IREE_TOKENIZER_TOKEN_ATTR_NONE);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Store score.
  if (ctx->current_index >= ctx->score_capacity) {
    // Grow scores array.
    iree_host_size_t new_capacity = ctx->score_capacity * 2;
    if (new_capacity < 1024) new_capacity = 1024;
    status = iree_allocator_realloc(
        ctx->allocator, new_capacity * sizeof(float), (void**)&ctx->scores);
    if (!iree_status_is_ok(status)) {
      ctx->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
    ctx->score_capacity = new_capacity;
  }
  ctx->scores[ctx->current_index++] = entry_ctx.score;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_import_unigram_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_vocab_t** out_vocab, float** out_scores,
    iree_host_size_t* out_score_count, float* out_unk_score) {
  IREE_ASSERT_ARGUMENT(out_vocab);
  IREE_ASSERT_ARGUMENT(out_scores);
  IREE_ASSERT_ARGUMENT(out_score_count);
  IREE_ASSERT_ARGUMENT(out_unk_score);
  *out_vocab = NULL;
  *out_scores = NULL;
  *out_score_count = 0;
  *out_unk_score = -10.0f;  // Default UNK penalty.
  IREE_TRACE_ZONE_BEGIN(z0);

  // Extract and validate the model object.
  iree_string_view_t model;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_json_extract_model(json, &model));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_json_validate_model_type(model, IREE_SV("Unigram")));

  // Validate model object keys.
  static const iree_string_view_t kModelAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("vocab"),
      IREE_SVL("unk_id"),
      IREE_SVL("byte_fallback"),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_validate_object_keys(model, kModelAllowedKeys,
                                         IREE_ARRAYSIZE(kModelAllowedKeys)));

  // Parse unk_id from model.
  int64_t unk_id = -1;
  iree_status_t status =
      iree_json_try_lookup_int64(model, IREE_SV("unk_id"), -1, &unk_id);

  // Parse added_tokens into temporary storage.
  iree_tokenizer_added_tokens_t added_tokens;
  iree_tokenizer_added_tokens_initialize(&added_tokens, allocator);
  if (iree_status_is_ok(status)) {
    status =
        iree_tokenizer_added_tokens_parse_json(json, allocator, &added_tokens);
  }

  // Get vocab array.
  iree_string_view_t vocab_array = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status =
        iree_json_lookup_object_value(model, IREE_SV("vocab"), &vocab_array);
  }

  // Count vocab entries for capacity.
  iree_host_size_t vocab_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_json_array_length(vocab_array, &vocab_count);
  }

  // Allocate builder.
  iree_host_size_t capacity = vocab_count + added_tokens.count;
  iree_tokenizer_vocab_builder_t* builder = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_tokenizer_vocab_builder_allocate(capacity, allocator, &builder);
  }

  // Allocate initial scores array.
  float* scores = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, capacity * sizeof(float),
                                   (void**)&scores);
  }

  // Parse vocab array.
  iree_tokenizer_parse_unigram_ctx_t parse_ctx = {
      .builder = builder,
      .scores = scores,
      .score_capacity = capacity,
      .current_index = 0,
      .allocator = allocator,
      .unescape_buffer = NULL,
      .unescape_capacity = 0,
      .status = iree_ok_status(),
  };

  if (iree_status_is_ok(status)) {
    status = iree_json_enumerate_array(
        vocab_array, iree_tokenizer_unigram_vocab_visitor, &parse_ctx);
    if (iree_status_is_ok(status) && !iree_status_is_ok(parse_ctx.status)) {
      status = parse_ctx.status;
    }
    scores = parse_ctx.scores;  // May have been reallocated.
  }

  // Free unescape buffer.
  if (parse_ctx.unescape_buffer) {
    iree_allocator_free(allocator, parse_ctx.unescape_buffer);
  }

  // Mark added_tokens entries that overlap with vocab as found_in_vocab.
  // Unigram vocab uses implicit sequential IDs (0, 1, 2...), so any
  // added_tokens with ID < vocab_count already exists in the vocab.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < added_tokens.count; ++i) {
      if ((iree_host_size_t)added_tokens.entries[i].id <
          parse_ctx.current_index) {
        added_tokens.entries[i].found_in_vocab = true;
      }
    }
  }

  // Add tokens from added_tokens and set special IDs.
  iree_host_size_t base_vocab_count = parse_ctx.current_index;
  if (iree_status_is_ok(status)) {
    if (added_tokens.count > 0) {
      status = iree_tokenizer_added_tokens_finalize(&added_tokens, builder);
    }
  }

  // Initialize scores for any added tokens that weren't in the base vocab.
  // These are typically special tokens that shouldn't be selected by Viterbi,
  // so we use the unk_score (very negative) to prevent their selection.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = base_vocab_count; i < parse_ctx.score_capacity;
         ++i) {
      scores[i] = *out_unk_score;
    }
  }

  // Override unk_id from JSON if specified.
  if (iree_status_is_ok(status) && unk_id >= 0) {
    status = iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, (int32_t)unk_id);
  }

  // Get UNK score if we have an UNK token.
  if (iree_status_is_ok(status) && unk_id >= 0 &&
      (iree_host_size_t)unk_id < parse_ctx.current_index) {
    *out_unk_score = scores[unk_id];
  }

  // Build vocab.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(builder, out_vocab);
    builder = NULL;  // Consumed by build.
  }

  // Ensure scores array matches vocab capacity.
  // Added tokens may have IDs beyond the initial capacity, requiring realloc.
  // If capacity is smaller (e.g., added_tokens were already in vocab), the
  // extra slots are harmless but we return the correct count.
  if (iree_status_is_ok(status)) {
    iree_host_size_t vocab_capacity = iree_tokenizer_vocab_capacity(*out_vocab);
    if (vocab_capacity > parse_ctx.score_capacity) {
      status = iree_allocator_realloc(allocator, vocab_capacity * sizeof(float),
                                      (void**)&scores);
      if (iree_status_is_ok(status)) {
        // Initialize new slots to unk_score.
        for (iree_host_size_t i = parse_ctx.score_capacity; i < vocab_capacity;
             ++i) {
          scores[i] = *out_unk_score;
        }
      }
    }
    if (iree_status_is_ok(status)) {
      *out_scores = scores;
      *out_score_count = vocab_capacity;  // Must match vocab for validation.
      scores = NULL;                      // Ownership transferred.
    }
  }

  // Cleanup on error.
  if (!iree_status_is_ok(status)) {
    if (builder) iree_tokenizer_vocab_builder_free(builder);
    if (scores) iree_allocator_free(allocator, scores);
  }
  iree_tokenizer_added_tokens_deinitialize(&added_tokens);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Unigram Tokenizer from JSON Factory
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_from_unigram_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Initialize transform, decoder, and postprocessor to safe defaults.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_none(&transform);
  iree_tokenizer_decoder_t decoder;
  iree_tokenizer_decoder_initialize_none(&decoder);
  iree_tokenizer_postprocessor_t postprocessor;
  iree_tokenizer_postprocessor_initialize_none(&postprocessor);

  // Import vocab and scores from JSON.
  iree_tokenizer_vocab_t* vocab = NULL;
  float* scores = NULL;
  iree_host_size_t score_count = 0;
  float unk_score = -10.0f;
  iree_status_t status = iree_tokenizer_vocab_import_unigram_json(
      json, allocator, &vocab, &scores, &score_count, &unk_score);

  // Parse pre-tokenizer transform from JSON (includes normalizer).
  if (iree_status_is_ok(status)) {
    status =
        iree_tokenizer_text_transform_parse_json(json, allocator, &transform);
  }

  // Parse decoder from JSON.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_decoder_parse_json(json, &decoder);
  }

  // Parse post_processor from JSON.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_postprocessor_parse_json(json, allocator,
                                                     &postprocessor);
  }

  // Create tokenizer with vocab and scores.
  // iree_tokenizer_unigram_allocate consumes vocab and scores.
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_unigram_allocate(vocab, scores, score_count,
                                             unk_score, allocator, &tokenizer);
  } else {
    if (vocab) iree_tokenizer_vocab_free(vocab);
    if (scores) iree_allocator_free(allocator, scores);
  }
  vocab = NULL;   // Consumed or freed.
  scores = NULL;  // Consumed or freed.

  // Parse added_tokens (literals) from JSON.
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
