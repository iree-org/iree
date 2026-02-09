// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/bpe_json.h"

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "iree/tokenizer/bpe.h"
#include "iree/tokenizer/huggingface/added_tokens.h"
#include "iree/tokenizer/huggingface/decoder_json.h"
#include "iree/tokenizer/huggingface/literals_json.h"
#include "iree/tokenizer/huggingface/postprocessor_json.h"
#include "iree/tokenizer/huggingface/transform_json.h"
#include "iree/tokenizer/huggingface/vocab_json.h"
#include "iree/tokenizer/vocab_builder.h"

//===----------------------------------------------------------------------===//
// Parse Merges (BPE-specific)
//===----------------------------------------------------------------------===//

typedef struct {
  iree_tokenizer_vocab_builder_t* builder;
  iree_tokenizer_vocab_t* vocab;  // Temporary vocab for lookups.
  iree_allocator_t allocator;
  char* unescape_buffer;
  iree_host_size_t unescape_capacity;
  iree_status_t status;
} iree_tokenizer_parse_merges_ctx_t;

// Context for parsing array-format merge elements: ["left", "right"].
typedef struct {
  iree_tokenizer_parse_merges_ctx_t* parent_ctx;
  iree_host_size_t element_count;  // Number of elements seen.
  int32_t left_id;
  int32_t right_id;
} iree_tokenizer_merge_array_ctx_t;

// Visitor for array-format merge elements.
static iree_status_t iree_tokenizer_bpe_parse_merge_array_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t value) {
  iree_tokenizer_merge_array_ctx_t* ctx =
      (iree_tokenizer_merge_array_ctx_t*)user_data;
  iree_tokenizer_parse_merges_ctx_t* parent = ctx->parent_ctx;
  ctx->element_count++;

  // Merges must have exactly 2 elements.
  if (index >= 2) {
    parent->status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "array-format merge must have exactly 2 "
                                      "elements, got more");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Unescape the token string.
  iree_host_size_t unescaped_length;
  iree_status_t status =
      iree_json_unescape_string(value, 0, NULL, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    parent->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  if (unescaped_length > parent->unescape_capacity) {
    status = iree_allocator_grow_array(
        parent->allocator, iree_max(256, unescaped_length), /*element_size=*/1,
        &parent->unescape_capacity, (void**)&parent->unescape_buffer);
    if (!iree_status_is_ok(status)) {
      parent->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
  }

  status =
      iree_json_unescape_string(value, parent->unescape_capacity,
                                parent->unescape_buffer, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    parent->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  iree_string_view_t token_str =
      iree_make_string_view(parent->unescape_buffer, unescaped_length);
  int32_t token_id = iree_tokenizer_vocab_lookup(parent->vocab, token_str);

  if (index == 0) {
    ctx->left_id = token_id;
  } else {
    ctx->right_id = token_id;
  }

  return iree_ok_status();
}

// Parses an array-format merge: ["left", "right"].
static iree_status_t iree_tokenizer_bpe_parse_merge_array(
    iree_tokenizer_parse_merges_ctx_t* ctx, iree_string_view_t array_value) {
  iree_tokenizer_merge_array_ctx_t array_ctx = {
      .parent_ctx = ctx,
      .element_count = 0,
      .left_id = -1,
      .right_id = -1,
  };

  iree_status_t status = iree_json_enumerate_array(
      array_value, iree_tokenizer_bpe_parse_merge_array_visitor, &array_ctx);

  // Check for errors during enumeration.
  if (!iree_status_is_ok(ctx->status)) {
    iree_status_ignore(status);
    return ctx->status;
  }
  if (iree_status_is_cancelled(status)) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(status);

  // Verify we got exactly 2 elements.
  if (array_ctx.element_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "array-format merge must have exactly 2 elements, "
                            "got %zu",
                            array_ctx.element_count);
  }

  // Verify both tokens were found.
  if (array_ctx.left_id < 0 || array_ctx.right_id < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "array-format merge references unknown token(s): "
                            "%.*s (left=%d, right=%d)",
                            (int)array_value.size, array_value.data,
                            (int)array_ctx.left_id, (int)array_ctx.right_id);
  }

  // Add the merge.
  return iree_tokenizer_vocab_builder_add_merge(
      ctx->builder, (uint32_t)array_ctx.left_id, (uint32_t)array_ctx.right_id);
}

// Parses a string-format merge: "left right".
static iree_status_t iree_tokenizer_bpe_parse_merge_string(
    iree_tokenizer_parse_merges_ctx_t* ctx, iree_string_view_t value) {
  // Unescape the merge string.
  iree_host_size_t unescaped_length;
  iree_status_t status =
      iree_json_unescape_string(value, 0, NULL, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (unescaped_length > ctx->unescape_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        ctx->allocator, iree_max(256, unescaped_length), /*element_size=*/1,
        &ctx->unescape_capacity, (void**)&ctx->unescape_buffer));
  }

  status = iree_json_unescape_string(value, ctx->unescape_capacity,
                                     ctx->unescape_buffer, &unescaped_length);
  IREE_RETURN_IF_ERROR(status);

  // Parse "left right" format - split on the first space.
  // This matches HuggingFace behavior: tokens with spaces must use array
  // format.
  iree_string_view_t merge_str =
      iree_make_string_view(ctx->unescape_buffer, unescaped_length);

  iree_host_size_t space_pos =
      iree_string_view_find_char(merge_str, ' ', /*pos=*/0);
  if (space_pos == IREE_STRING_VIEW_NPOS || space_pos == 0 ||
      space_pos == merge_str.size - 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "merge must have format \"left right\": \"%.*s\"",
                            (int)merge_str.size, merge_str.data);
  }

  iree_string_view_t left = iree_string_view_substr(merge_str, 0, space_pos);
  iree_string_view_t right =
      iree_string_view_substr(merge_str, space_pos + 1, IREE_HOST_SIZE_MAX);

  int32_t left_id = iree_tokenizer_vocab_lookup(ctx->vocab, left);
  int32_t right_id = iree_tokenizer_vocab_lookup(ctx->vocab, right);

  if (left_id < 0 || right_id < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "merge rule references unknown token(s): \"%.*s\" (left=%d, right=%d)",
        (int)merge_str.size, merge_str.data, (int)left_id, (int)right_id);
  }

  return iree_tokenizer_vocab_builder_add_merge(ctx->builder, (uint32_t)left_id,
                                                (uint32_t)right_id);
}

static iree_status_t iree_tokenizer_bpe_parse_merges_visitor(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  (void)index;
  iree_tokenizer_parse_merges_ctx_t* ctx =
      (iree_tokenizer_parse_merges_ctx_t*)user_data;

  // Use JSON type to distinguish merge formats:
  // - IREE_JSON_VALUE_TYPE_ARRAY: ["left", "right"] - modern HuggingFace style
  // - IREE_JSON_VALUE_TYPE_STRING: "left right" - legacy style
  // Type is inferred before quote stripping, so string "[ i" is correctly
  // identified as STRING type (not confused with array format).
  iree_status_t status;
  if (type == IREE_JSON_VALUE_TYPE_ARRAY) {
    status = iree_tokenizer_bpe_parse_merge_array(ctx, value);
  } else if (type == IREE_JSON_VALUE_TYPE_STRING) {
    status = iree_tokenizer_bpe_parse_merge_string(ctx, value);
  } else {
    ctx->status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                   "merge must be array or string");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  return iree_ok_status();
}

// Maximum number of merges allowed to prevent memory exhaustion from malicious
// JSON. 256k covers all real models (GPT-2 has ~50k, LLaMA has ~128k).
#define IREE_TOKENIZER_MAX_MERGES 262144

static iree_status_t iree_tokenizer_bpe_json_parse_merges(
    iree_string_view_t model, iree_allocator_t allocator,
    iree_tokenizer_vocab_builder_t* builder, iree_tokenizer_vocab_t* vocab) {
  iree_string_view_t merges;
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_object_value(model, IREE_SV("merges"), &merges));
  if (merges.size == 0) {
    return iree_ok_status();  // No merges, that's fine.
  }

  // Count merges and validate against limits before allocating.
  iree_host_size_t merge_count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(merges, &merge_count));
  if (merge_count > IREE_TOKENIZER_MAX_MERGES) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "merge count %" PRIhsz
                            " exceeds maximum allowed (%d)",
                            merge_count, IREE_TOKENIZER_MAX_MERGES);
  }

  iree_tokenizer_parse_merges_ctx_t ctx = {
      .builder = builder,
      .vocab = vocab,
      .allocator = allocator,
      .unescape_buffer = NULL,
      .unescape_capacity = 0,
      .status = iree_ok_status(),
  };

  iree_status_t status = iree_json_enumerate_array_typed(
      merges, iree_tokenizer_bpe_parse_merges_visitor, &ctx);

  if (ctx.unescape_buffer) {
    iree_allocator_free(allocator, ctx.unescape_buffer);
  }

  if (!iree_status_is_ok(ctx.status)) {
    iree_status_ignore(status);
    return ctx.status;
  }
  if (iree_status_is_cancelled(status)) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_import_bpe_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_vocab_t** out_vocab) {
  IREE_ASSERT_ARGUMENT(out_vocab);
  *out_vocab = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Extract and validate the model object.
  iree_string_view_t model;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_json_extract_model(json, &model));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_json_validate_model_type(model, IREE_SV("BPE")));

  // Validate model object keys.
  static const iree_string_view_t kModelAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("vocab"),
      IREE_SVL("merges"),
      IREE_SVL("end_of_word_suffix"),
      IREE_SVL("unk_token"),
      IREE_SVL("fuse_unk"),
      IREE_SVL("byte_fallback"),
      IREE_SVL("dropout"),
      IREE_SVL("continuing_subword_prefix"),
      IREE_SVL("ignore_merges"),
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

  // Phase 3: Parse vocab into builder (checking against added_tokens for
  // attrs).
  iree_tokenizer_vocab_builder_t* builder = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_tokenizer_vocab_builder_allocate(capacity, allocator, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_json_parse_vocab(model, allocator, &added_tokens,
                                             builder);
  }

  // Build a temporary vocab for merge token lookups.
  // We need the hash table to resolve merge strings to token IDs.
  iree_tokenizer_vocab_t* temp_vocab = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(builder, &temp_vocab);
    builder = NULL;  // Builder consumed by build().
  }

  // Create final builder with merges.
  iree_tokenizer_vocab_builder_t* final_builder = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_allocate(capacity, allocator,
                                                   &final_builder);
  }

  // Re-add all tokens from temp_vocab to final_builder (preserving IDs/attrs).
  // IMPORTANT: Must use add_token_with_id to preserve original IDs, otherwise
  // merge rules (which reference original IDs) will be broken.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < iree_tokenizer_vocab_capacity(temp_vocab);
         ++i) {
      iree_tokenizer_token_attr_t attrs =
          iree_tokenizer_vocab_token_attrs(temp_vocab, (int32_t)i);
      // Skip gap tokens (sparse vocabs have UNUSED placeholder entries).
      if (iree_all_bits_set(attrs, IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
        continue;
      }
      iree_string_view_t text =
          iree_tokenizer_vocab_token_text(temp_vocab, (int32_t)i);
      float score = iree_tokenizer_vocab_token_score(temp_vocab, (int32_t)i);
      status = iree_tokenizer_vocab_builder_add_token_with_id(
          final_builder, (int32_t)i, text, score, attrs);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Phase 4: Parse merges.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_bpe_json_parse_merges(model, allocator,
                                                  final_builder, temp_vocab);
  }

  // Free temp vocab (no longer needed after merges parsed).
  if (temp_vocab) {
    iree_tokenizer_vocab_free(temp_vocab);
    temp_vocab = NULL;
  }

  // Phase 5: Add missing tokens and set special IDs from added_tokens,
  // OR use fallback detection if added_tokens was empty.
  if (iree_status_is_ok(status)) {
    if (added_tokens.count > 0) {
      status =
          iree_tokenizer_added_tokens_finalize(&added_tokens, final_builder);
    } else {
      status = iree_tokenizer_detect_specials_from_vocab(model, final_builder);
    }
  }

  // Build final vocab on success, cleanup on failure.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(final_builder, out_vocab);
  } else {
    if (builder) iree_tokenizer_vocab_builder_free(builder);
    if (final_builder) iree_tokenizer_vocab_builder_free(final_builder);
  }
  iree_tokenizer_added_tokens_deinitialize(&added_tokens);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// BPE Tokenizer from JSON Factory
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_from_bpe_json(iree_string_view_t json,
                                           iree_allocator_t allocator,
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
      iree_tokenizer_vocab_import_bpe_json(json, allocator, &vocab);

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
  // This determines whether BOS/EOS/CLS/SEP are added when encoding.
  // A null/missing post_processor means no special tokens are added (GPT-2).
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_postprocessor_parse_json(json, allocator,
                                                     &postprocessor);
  }

  // Create tokenizer with vocab. Allocate consumes vocab unconditionally
  // (frees on failure, owns on success) - caller must not use vocab after.
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_bpe_allocate(vocab, allocator, &tokenizer);
  } else {
    iree_tokenizer_vocab_free(vocab);
  }
  vocab = NULL;  // Consumed by allocate or freed above.

  // Parse and set end_of_word_suffix (e.g., "</w>" for CLIP).
  // This suffix is appended to each word before BPE tokenization.
  if (iree_status_is_ok(status)) {
    iree_string_view_t model;
    status = iree_tokenizer_json_extract_model(json, &model);
    if (iree_status_is_ok(status)) {
      char suffix_storage[16];  // Must match bpe.c state storage.
      iree_host_size_t suffix_length = 0;
      // Empty default means no suffix if key is missing or null.
      status = iree_json_try_lookup_string(
          model, IREE_SV("end_of_word_suffix"), iree_string_view_empty(),
          iree_make_mutable_string_view(suffix_storage, sizeof(suffix_storage)),
          &suffix_length);
      if (iree_status_is_ok(status) && suffix_length > 0) {
        status = iree_tokenizer_bpe_set_end_of_word_suffix(
            tokenizer, iree_make_string_view(suffix_storage, suffix_length));
      }
    }
  }

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
    // Cleanup on failure (vocab already freed above).
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
