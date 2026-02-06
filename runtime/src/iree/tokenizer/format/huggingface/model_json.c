// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/model_json.h"

#include "iree/base/internal/json.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/model/unigram.h"
#include "iree/tokenizer/model/wordpiece.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

//===----------------------------------------------------------------------===//
// Internal Constants and Helpers
//===----------------------------------------------------------------------===//

// Maximum token length for stack-allocated buffers during merge parsing.
// Empirical analysis of 170+ tokenizer files shows:
//   - Max token length observed: 512 bytes (256 Ġ chars in EleutherAI models)
//   - Max merge string: 1025 bytes (512 + space + 512 for pythia/gpt-neox)
//   - >99.99% of tokens are under 128 bytes
// Use 2048 to provide headroom for future tokenizers.
#define IREE_TOKENIZER_MAX_MERGE_TOKEN_LENGTH 2048

// Model type for internal dispatch. Auto-detected from JSON.
typedef enum iree_tokenizer_model_type_e {
  IREE_TOKENIZER_MODEL_TYPE_UNKNOWN = 0,
  IREE_TOKENIZER_MODEL_TYPE_BPE,
  IREE_TOKENIZER_MODEL_TYPE_WORDPIECE,
  IREE_TOKENIZER_MODEL_TYPE_UNIGRAM,
} iree_tokenizer_model_type_t;

// Reference: tokenizers/src/models/bpe/model.rs (BPE struct fields)
static const iree_string_view_t kBPEAllowedKeys[] = {
    IREE_SVL("type"),                       // "BPE"
    IREE_SVL("vocab"),                      // {token: id, ...}
    IREE_SVL("merges"),                     // ["a b", ...] or [["a","b"], ...]
    IREE_SVL("dropout"),                    // float|null (training only)
    IREE_SVL("unk_token"),                  // string|null
    IREE_SVL("continuing_subword_prefix"),  // string|null
    IREE_SVL("end_of_word_suffix"),         // string|null (CLIP uses "</w>")
    IREE_SVL("fuse_unk"),                   // bool (default false)
    IREE_SVL("byte_fallback"),              // bool (default false)
    IREE_SVL("ignore_merges"),              // bool (default false)
};

// Reference: tokenizers/src/models/wordpiece/serialization.rs
static const iree_string_view_t kWordPieceAllowedKeys[] = {
    IREE_SVL("type"),                       // "WordPiece"
    IREE_SVL("vocab"),                      // {token: id, ...}
    IREE_SVL("unk_token"),                  // string (default "[UNK]")
    IREE_SVL("continuing_subword_prefix"),  // string (default "##")
    IREE_SVL("max_input_chars_per_word"),   // integer (default 100)
};

// Reference: tokenizers/src/models/unigram/serialization.rs
static const iree_string_view_t kUnigramAllowedKeys[] = {
    IREE_SVL("type"),           // "Unigram"
    IREE_SVL("vocab"),          // [[token, score], ...]
    IREE_SVL("unk_id"),         // integer|null (index in vocab array)
    IREE_SVL("byte_fallback"),  // bool (default true for SentencePiece)
};

// Unescapes a JSON string into a fixed-size buffer, returning a view of the
// result. The returned string_view points into buffer and is valid until the
// next call. Fails if the unescaped result exceeds buffer_capacity.
static iree_status_t iree_tokenizer_model_json_unescape(
    iree_string_view_t input, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_unescaped) {
  *out_unescaped = iree_string_view_empty();
  iree_host_size_t unescaped_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(input, buffer_capacity, buffer,
                                                 &unescaped_length));
  *out_unescaped = iree_make_string_view(buffer, unescaped_length);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Shared Parsing Helpers
//===----------------------------------------------------------------------===//

// Context for vocab object enumeration.
typedef struct iree_tokenizer_vocab_parse_context_t {
  iree_tokenizer_vocab_builder_t* builder;
  char unescape_buffer[IREE_TOKENIZER_MAX_MERGE_TOKEN_LENGTH];
  iree_status_t status;
} iree_tokenizer_vocab_parse_context_t;

// Visitor callback for parsing vocab object entries.
static iree_status_t iree_tokenizer_vocab_entry_visitor(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  iree_tokenizer_vocab_parse_context_t* context =
      (iree_tokenizer_vocab_parse_context_t*)user_data;

  // Unescape the key (token text).
  iree_string_view_t token_text = iree_string_view_empty();
  iree_status_t status = iree_tokenizer_model_json_unescape(
      key, context->unescape_buffer, sizeof(context->unescape_buffer),
      &token_text);

  // Parse the value (token ID).
  int64_t token_id_raw = 0;
  if (iree_status_is_ok(status)) {
    status = iree_json_parse_int64(value, &token_id_raw);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "invalid token ID for '%.*s'",
                                      (int)token_text.size, token_text.data);
    }
  }

  // Validate ID range.
  if (iree_status_is_ok(status)) {
    if (token_id_raw < 0 || token_id_raw > INT32_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "token ID %" PRId64 " for '%.*s' out of valid range [0, %" PRId32 "]",
          token_id_raw, (int)token_text.size, token_text.data, INT32_MAX);
    }
  }

  // Add token to builder.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_add_token_with_id(
        context->builder, (int32_t)token_id_raw, token_text, 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE);
  }

  if (!iree_status_is_ok(status)) {
    context->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return iree_ok_status();
}

// Parses vocab object into a vocab builder.
static iree_status_t iree_tokenizer_parse_vocab_object(
    iree_string_view_t vocab_json, iree_tokenizer_vocab_builder_t* builder,
    iree_allocator_t allocator) {
  (void)allocator;  // Unused after removing dynamic buffer.
  iree_tokenizer_vocab_parse_context_t context = {
      .builder = builder,
      .unescape_buffer = {0},
      .status = iree_ok_status(),
  };

  iree_status_t status = iree_json_enumerate_object(
      vocab_json, iree_tokenizer_vocab_entry_visitor, &context);

  // Check visitor status first (contains detailed error).
  if (!iree_status_is_ok(context.status)) {
    if (iree_status_is_ok(status) ||
        iree_status_code(status) == IREE_STATUS_CANCELLED) {
      iree_status_ignore(status);
      return context.status;
    }
    // Both have errors - prefer visitor's (more specific).
    iree_status_ignore(status);
    return context.status;
  }

  // Handle CANCELLED from early stop.
  if (iree_status_code(status) == IREE_STATUS_CANCELLED) {
    iree_status_ignore(status);
    return iree_ok_status();
  }

  return status;
}

// Context for resolving a token text to its ID via linear scan of vocab JSON.
typedef struct iree_tokenizer_token_resolve_context_t {
  iree_string_view_t target_text;
  char unescape_buffer[IREE_TOKENIZER_MAX_MERGE_TOKEN_LENGTH];
  iree_tokenizer_token_id_t resolved_id;
  iree_status_t status;
} iree_tokenizer_token_resolve_context_t;

// Visitor callback that checks each vocab entry against the target text.
static iree_status_t iree_tokenizer_token_resolve_visitor(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  iree_tokenizer_token_resolve_context_t* context =
      (iree_tokenizer_token_resolve_context_t*)user_data;

  // Check if token could be too long. JSON escapes only make strings longer
  // (e.g., " → \", © → \u00A9), so unescaped length <= escaped length.
  // If the escaped form doesn't fit, query the actual unescaped length.
  if (key.size > sizeof(context->unescape_buffer)) {
    iree_host_size_t required_length = 0;
    iree_status_t status =
        iree_json_unescape_string(key, 0, NULL, &required_length);
    if (!iree_status_is_ok(status)) {
      context->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
    if (required_length > sizeof(context->unescape_buffer)) {
      // Token too long to be our target - skip.
      return iree_ok_status();
    }
  }

  // Unescape the key (token text). This cannot fail with RESOURCE_EXHAUSTED
  // because we already verified the length above.
  iree_host_size_t unescaped_length = 0;
  iree_status_t status =
      iree_json_unescape_string(key, sizeof(context->unescape_buffer),
                                context->unescape_buffer, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    context->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  iree_string_view_t token_text =
      iree_make_string_view(context->unescape_buffer, unescaped_length);

  // Check if this is the token we're looking for.
  if (!iree_string_view_equal(token_text, context->target_text)) {
    return iree_ok_status();
  }

  // Parse the value (token ID).
  int64_t token_id_raw = 0;
  status = iree_json_parse_int64(value, &token_id_raw);
  if (!iree_status_is_ok(status)) {
    context->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (token_id_raw < 0 || token_id_raw > INT32_MAX) {
    context->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                       "resolved token ID %" PRId64
                                       " out of valid range [0, %" PRId32 "]",
                                       token_id_raw, INT32_MAX);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  context->resolved_id = (iree_tokenizer_token_id_t)token_id_raw;
  return iree_status_from_code(IREE_STATUS_CANCELLED);  // Found, stop.
}

// Scans a vocab JSON object for a token matching |target_text| and returns its
// ID. This is a linear scan that processes each entry without building any
// index structure — suitable for resolving a small number of known tokens.
// Returns NOT_FOUND if the token is not present in the vocab.
static iree_status_t iree_tokenizer_resolve_token_in_vocab_json(
    iree_string_view_t vocab_json, iree_string_view_t target_text,
    iree_tokenizer_token_id_t* out_token_id) {
  *out_token_id = IREE_TOKENIZER_TOKEN_ID_INVALID;

  iree_tokenizer_token_resolve_context_t context = {
      .target_text = target_text,
      .unescape_buffer = {0},
      .resolved_id = IREE_TOKENIZER_TOKEN_ID_INVALID,
      .status = iree_ok_status(),
  };

  iree_status_t status = iree_json_enumerate_object(
      vocab_json, iree_tokenizer_token_resolve_visitor, &context);

  // Check visitor error first (more specific than enumeration error).
  if (!iree_status_is_ok(context.status)) {
    iree_status_ignore(status);
    return context.status;
  }
  IREE_RETURN_IF_ERROR(status);

  // Check if the token was found (visitor sets resolved_id on match).
  if (context.resolved_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    *out_token_id = context.resolved_id;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "token '%.*s' not found in vocabulary",
                          (int)target_text.size, target_text.data);
}

// Context for merge array enumeration.
typedef struct iree_tokenizer_merge_parse_context_t {
  iree_tokenizer_vocab_builder_t* builder;
  const iree_tokenizer_vocab_t* vocab;
  char unescape_buffer[IREE_TOKENIZER_MAX_MERGE_TOKEN_LENGTH];
  iree_status_t status;
} iree_tokenizer_merge_parse_context_t;

// Parses a merge in "left right" string format.
static iree_status_t iree_tokenizer_parse_merge_string(
    iree_tokenizer_merge_parse_context_t* context, iree_host_size_t index,
    iree_string_view_t merge_str) {
  // Unescape the merge string.
  iree_string_view_t unescaped = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_tokenizer_model_json_unescape(
      merge_str, context->unescape_buffer, sizeof(context->unescape_buffer),
      &unescaped));

  // Find the space separator between left and right tokens.
  iree_host_size_t space_pos = iree_string_view_find_char(unescaped, ' ', 0);
  if (space_pos == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "merge at index %" PRIhsz
                            " has invalid format '%.*s' (expected "
                            "'left right')",
                            index, (int)unescaped.size, unescaped.data);
  }

  iree_string_view_t left_text =
      iree_string_view_substr(unescaped, 0, space_pos);
  iree_string_view_t right_text = iree_string_view_substr(
      unescaped, space_pos + 1, unescaped.size - space_pos - 1);

  // Look up token IDs.
  iree_tokenizer_token_id_t left_id =
      iree_tokenizer_vocab_lookup(context->vocab, left_text);
  if (left_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "merge at index %" PRIhsz
                            " references unknown left token '%.*s'",
                            index, (int)left_text.size, left_text.data);
  }

  iree_tokenizer_token_id_t right_id =
      iree_tokenizer_vocab_lookup(context->vocab, right_text);
  if (right_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "merge at index %" PRIhsz
                            " references unknown right token '%.*s'",
                            index, (int)right_text.size, right_text.data);
  }

  return iree_tokenizer_vocab_builder_add_merge(
      context->builder, (uint32_t)left_id, (uint32_t)right_id);
}

// Parses a merge in ["left", "right"] tuple format.
static iree_status_t iree_tokenizer_parse_merge_tuple(
    iree_tokenizer_merge_parse_context_t* context, iree_host_size_t index,
    iree_string_view_t merge_array) {
  // Get the two elements.
  iree_string_view_t left_json = iree_string_view_empty();
  iree_string_view_t right_json = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_array_get(merge_array, 0, &left_json));
  IREE_RETURN_IF_ERROR(iree_json_array_get(merge_array, 1, &right_json));

  // Unescape left token into context buffer, then copy to local.
  iree_string_view_t left_text = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_tokenizer_model_json_unescape(
      left_json, context->unescape_buffer, sizeof(context->unescape_buffer),
      &left_text));

  // Copy left_text since unescape buffer will be reused for right token.
  char left_copy[IREE_TOKENIZER_MAX_MERGE_TOKEN_LENGTH];
  memcpy(left_copy, left_text.data, left_text.size);
  left_text = iree_make_string_view(left_copy, left_text.size);

  // Unescape right token (reuses buffer).
  iree_string_view_t right_text = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_tokenizer_model_json_unescape(
      right_json, context->unescape_buffer, sizeof(context->unescape_buffer),
      &right_text));

  // Look up token IDs.
  iree_tokenizer_token_id_t left_id =
      iree_tokenizer_vocab_lookup(context->vocab, left_text);
  if (left_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "merge at index %" PRIhsz
                            " references unknown left token '%.*s'",
                            index, (int)left_text.size, left_text.data);
  }

  iree_tokenizer_token_id_t right_id =
      iree_tokenizer_vocab_lookup(context->vocab, right_text);
  if (right_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "merge at index %" PRIhsz
                            " references unknown right token '%.*s'",
                            index, (int)right_text.size, right_text.data);
  }

  return iree_tokenizer_vocab_builder_add_merge(
      context->builder, (uint32_t)left_id, (uint32_t)right_id);
}

// Visitor callback for parsing merge array elements (typed version).
static iree_status_t iree_tokenizer_merge_visitor(void* user_data,
                                                  iree_host_size_t index,
                                                  iree_json_value_type_t type,
                                                  iree_string_view_t value) {
  iree_tokenizer_merge_parse_context_t* context =
      (iree_tokenizer_merge_parse_context_t*)user_data;

  iree_status_t status = iree_ok_status();
  if (type == IREE_JSON_VALUE_TYPE_STRING) {
    status = iree_tokenizer_parse_merge_string(context, index, value);
  } else if (type == IREE_JSON_VALUE_TYPE_ARRAY) {
    status = iree_tokenizer_parse_merge_tuple(context, index, value);
  } else {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "merge at index %" PRIhsz
                              " has unexpected type (expected string or array)",
                              index);
  }

  if (!iree_status_is_ok(status)) {
    context->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return iree_ok_status();
}

// Parses merges array into a vocab builder.
static iree_status_t iree_tokenizer_parse_merges_array(
    iree_string_view_t merges_json, iree_tokenizer_vocab_builder_t* builder,
    const iree_tokenizer_vocab_t* vocab, iree_allocator_t allocator) {
  (void)allocator;  // Unused after removing dynamic buffer.
  iree_tokenizer_merge_parse_context_t context = {
      .builder = builder,
      .vocab = vocab,
      .unescape_buffer = {0},
      .status = iree_ok_status(),
  };

  iree_status_t status = iree_json_enumerate_array_typed(
      merges_json, iree_tokenizer_merge_visitor, &context);

  // Check visitor status first.
  if (!iree_status_is_ok(context.status)) {
    if (iree_status_is_ok(status) ||
        iree_status_code(status) == IREE_STATUS_CANCELLED) {
      iree_status_ignore(status);
      return context.status;
    }
    iree_status_ignore(status);
    return context.status;
  }

  if (iree_status_code(status) == IREE_STATUS_CANCELLED) {
    iree_status_ignore(status);
    return iree_ok_status();
  }

  return status;
}

//===----------------------------------------------------------------------===//
// BPE Model Parser
//===----------------------------------------------------------------------===//

// Maximum pre-allocation hint for vocabulary capacity. This clamps the initial
// allocation size to prevent a pathological JSON input from requesting an
// unreasonable amount of memory upfront. The builder still grows dynamically
// beyond this if the actual token count is higher.
// 2^22 = 4,194,304 tokens — 16x the largest real-world tokenizer (Gemma-3 at
// 262,144 tokens).
#define IREE_TOKENIZER_MAX_VOCAB_CAPACITY_HINT (4 * 1024 * 1024)

iree_host_size_t iree_tokenizer_huggingface_estimate_vocab_capacity(
    iree_host_size_t json_size) {
  // Heuristic: ~40 bytes per vocab entry on average.
  iree_host_size_t estimate = json_size / 40;
  estimate = iree_max(estimate, (iree_host_size_t)1000);
  if (estimate > IREE_TOKENIZER_MAX_VOCAB_CAPACITY_HINT) {
    return IREE_TOKENIZER_MAX_VOCAB_CAPACITY_HINT;
  }
  return estimate;
}

iree_status_t iree_tokenizer_huggingface_parse_bpe_model(
    iree_string_view_t model_json,
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_tokenizer_bpe_flags_t extra_flags, iree_allocator_t allocator,
    iree_tokenizer_model_t** out_model, iree_tokenizer_vocab_t** out_vocab) {
  IREE_ASSERT_ARGUMENT(out_model);
  IREE_ASSERT_ARGUMENT(out_vocab);
  *out_model = NULL;
  *out_vocab = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_validate_object_keys(model_json, kBPEAllowedKeys,
                                     IREE_ARRAYSIZE(kBPEAllowedKeys)),
      "in BPE model object");

  // Look up required fields.
  iree_string_view_t vocab_json = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_lookup_object_value(model_json, IREE_SV("vocab"), &vocab_json),
      "BPE model missing vocab");

  iree_string_view_t merges_json = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_lookup_object_value(model_json, IREE_SV("merges"),
                                    &merges_json),
      "BPE model missing merges");

  // Parse optional flags.
  bool byte_fallback = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_bool(model_json, IREE_SV("byte_fallback"), false,
                                    &byte_fallback));

  bool fuse_unk = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_bool(model_json, IREE_SV("fuse_unk"), false,
                                    &fuse_unk));

  bool ignore_merges = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_bool(model_json, IREE_SV("ignore_merges"), false,
                                    &ignore_merges));

  // Reject features that affect merge/word computation which we do not
  // implement. continuing_subword_prefix prepends a prefix (e.g. "##") to all
  // non-initial subword tokens and strips it during merge result lookup.
  // end_of_word_suffix appends a suffix (e.g. "</w>") to the last character in
  // each word. Both fundamentally change tokenization output if non-null.

  iree_string_view_t csp_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(
              model_json, IREE_SV("continuing_subword_prefix"), &csp_value));
  if (csp_value.size > 0 &&
      !iree_string_view_equal(csp_value, IREE_SV("null"))) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "BPE continuing_subword_prefix is not supported (value: %.*s)",
                (int)csp_value.size, csp_value.data));
  }

  iree_string_view_t eow_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(
              model_json, IREE_SV("end_of_word_suffix"), &eow_value));

  // Parse optional unk_token string for unknown byte fallback.
  iree_string_view_t unk_token_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(model_json, IREE_SV("unk_token"),
                                            &unk_token_value));

  // Resources that need cleanup from this point forward.
  iree_tokenizer_vocab_builder_t* builder = NULL;
  iree_tokenizer_vocab_t* temp_vocab = NULL;
  iree_tokenizer_vocab_t* vocab = NULL;
  iree_tokenizer_model_t* model = NULL;

  // Estimate capacity and create builder.
  iree_host_size_t capacity =
      iree_tokenizer_huggingface_estimate_vocab_capacity(model_json.size);
  iree_status_t status =
      iree_tokenizer_vocab_builder_allocate(capacity, allocator, &builder);

  // Pass 1: Parse vocab into builder.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_parse_vocab_object(vocab_json, builder, allocator);
  }

  // Build temporary vocab for merge token lookup.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(builder, &temp_vocab);
    if (iree_status_is_ok(status)) {
      builder = NULL;  // Consumed by build on success.
    }
  }

  // Resolve unk_token string to ID in temp_vocab. This is done before
  // temp_vocab is freed so we can look up the token by its text content.
  iree_tokenizer_token_id_t unk_token_id = IREE_TOKENIZER_TOKEN_ID_INVALID;
  if (iree_status_is_ok(status) && unk_token_value.size > 0 &&
      !iree_string_view_equal(unk_token_value, IREE_SV("null"))) {
    char unk_buffer[64];
    iree_host_size_t unk_length = 0;
    status = iree_json_unescape_string(unk_token_value, sizeof(unk_buffer),
                                       unk_buffer, &unk_length);
    if (iree_status_is_ok(status)) {
      unk_token_id = iree_tokenizer_vocab_lookup(
          temp_vocab, iree_make_string_view(unk_buffer, unk_length));
      if (unk_token_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "unk_token '%.*s' not found in vocabulary",
                                  (int)unk_length, unk_buffer);
      }
    }
  }

  // Create new builder for second pass with merges.
  if (iree_status_is_ok(status)) {
    status =
        iree_tokenizer_vocab_builder_allocate(capacity, allocator, &builder);
  }

  // Pass 2: Parse vocab again (needed because first builder was consumed).
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_parse_vocab_object(vocab_json, builder, allocator);
  }

  // Parse merges using temp_vocab for ID lookup.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_parse_merges_array(merges_json, builder, temp_vocab,
                                               allocator);
  }

  // Done with temp vocab.
  iree_tokenizer_vocab_free(temp_vocab);
  temp_vocab = NULL;

  // Process added_tokens: insert new tokens or update attributes on existing.
  if (iree_status_is_ok(status) && added_tokens) {
    for (iree_host_size_t i = 0;
         i < added_tokens->count && iree_status_is_ok(status); ++i) {
      const iree_tokenizer_huggingface_added_token_t* at =
          &added_tokens->tokens[i];

      // Map HuggingFace flags to vocab attributes.
      iree_tokenizer_token_attr_t attrs = IREE_TOKENIZER_TOKEN_ATTR_NONE;
      if (at->flags & IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL) {
        attrs |= IREE_TOKENIZER_TOKEN_ATTR_SPECIAL;
      }

      iree_string_view_t content =
          iree_tokenizer_huggingface_added_token_content(added_tokens, at);
      status = iree_tokenizer_vocab_builder_ensure_token(builder, at->id,
                                                         content, 0.0f, attrs);
    }
  }

  // Set unk_token special ID on the builder (resolved earlier from temp_vocab).
  if (iree_status_is_ok(status) &&
      unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    status = iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, (int32_t)unk_token_id);
  }

  // Build final vocab.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(builder, &vocab);
    if (iree_status_is_ok(status)) {
      builder = NULL;  // Consumed by build on success.
    }
  }

  // Map flags from JSON and merge with extra_flags from caller.
  iree_tokenizer_bpe_flags_t flags = extra_flags;
  if (!byte_fallback) {
    flags |= IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK;
  }
  if (fuse_unk) {
    flags |= IREE_TOKENIZER_BPE_FLAG_FUSE_UNK;
  }
  if (ignore_merges) {
    flags |= IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES;
  }

  // Create BPE model.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_bpe_model_allocate(vocab, flags, allocator, &model);
  }

  // Set end_of_word_suffix if specified (e.g., "</w>" for CLIP).
  if (iree_status_is_ok(status) && eow_value.size > 0 &&
      !iree_string_view_equal(eow_value, IREE_SV("null"))) {
    char eow_buffer[64];
    iree_host_size_t eow_length = 0;
    status = iree_json_unescape_string(eow_value, sizeof(eow_buffer),
                                       eow_buffer, &eow_length);
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_bpe_model_set_end_of_word_suffix(
          model, iree_make_string_view(eow_buffer, eow_length));
    }
  }

  if (iree_status_is_ok(status)) {
    *out_model = model;
    *out_vocab = vocab;
  } else {
    iree_tokenizer_model_free(model);
    iree_tokenizer_vocab_free(vocab);
    iree_tokenizer_vocab_builder_free(builder);
    iree_tokenizer_vocab_free(temp_vocab);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// WordPiece Model Parser
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_huggingface_parse_wordpiece_model(
    iree_string_view_t model_json,
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model,
    iree_tokenizer_vocab_t** out_vocab) {
  IREE_ASSERT_ARGUMENT(out_model);
  IREE_ASSERT_ARGUMENT(out_vocab);
  *out_model = NULL;
  *out_vocab = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_validate_object_keys(model_json, kWordPieceAllowedKeys,
                                     IREE_ARRAYSIZE(kWordPieceAllowedKeys)),
      "in WordPiece model object");

  // Look up required fields.
  iree_string_view_t vocab_json = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_lookup_object_value(model_json, IREE_SV("vocab"), &vocab_json),
      "WordPiece model missing vocab");

  // Parse unk_token: not present -> default "[UNK]", null -> disabled, else use
  // the string value.
  iree_string_view_t unk_token_raw = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(model_json, IREE_SV("unk_token"),
                                            &unk_token_raw));
  char unk_token_buffer[64];
  iree_host_size_t unk_token_length = 0;
  if (unk_token_raw.size == 0) {
    // Not present - use default "[UNK]".
    memcpy(unk_token_buffer, "[UNK]", 5);
    unk_token_length = 5;
  } else if (!iree_string_view_equal(unk_token_raw, IREE_SV("null"))) {
    // String value - unescape it.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_json_unescape_string(unk_token_raw, sizeof(unk_token_buffer),
                                      unk_token_buffer, &unk_token_length));
  }
  // else: null -> unk_token_length remains 0 (disabled).

  char prefix_buffer[16];
  iree_host_size_t prefix_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_try_lookup_string(
          model_json, IREE_SV("continuing_subword_prefix"), IREE_SV("##"),
          iree_make_mutable_string_view(prefix_buffer, sizeof(prefix_buffer)),
          &prefix_length));

  int64_t max_input_chars_per_word = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_int64(model_json,
                                     IREE_SV("max_input_chars_per_word"), 100,
                                     &max_input_chars_per_word));
  if (max_input_chars_per_word <= 0) {
    IREE_RETURN_AND_END_ZONE(
        z0,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "max_input_chars_per_word must be > 0, got %" PRId64,
                         max_input_chars_per_word));
  }

  // Resolve unk_token text to its ID via linear scan of the vocab JSON.
  // This avoids building a temporary vocab just for one lookup.
  iree_tokenizer_token_id_t unk_token_id = IREE_TOKENIZER_TOKEN_ID_INVALID;
  if (unk_token_length > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_tokenizer_resolve_token_in_vocab_json(
            vocab_json,
            iree_make_string_view(unk_token_buffer, unk_token_length),
            &unk_token_id),
        "resolving WordPiece unk_token");
  }

  // Resources that need cleanup from this point forward.
  iree_tokenizer_vocab_builder_t* builder = NULL;
  iree_tokenizer_vocab_t* vocab = NULL;
  iree_tokenizer_model_t* model = NULL;

  // Estimate capacity and create builder.
  iree_host_size_t capacity =
      iree_tokenizer_huggingface_estimate_vocab_capacity(model_json.size);
  iree_status_t status =
      iree_tokenizer_vocab_builder_allocate(capacity, allocator, &builder);

  // Parse vocab into builder.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_parse_vocab_object(vocab_json, builder, allocator);
  }

  // Process added_tokens: insert new tokens or update attributes on existing.
  if (iree_status_is_ok(status) && added_tokens) {
    for (iree_host_size_t i = 0;
         i < added_tokens->count && iree_status_is_ok(status); ++i) {
      const iree_tokenizer_huggingface_added_token_t* at =
          &added_tokens->tokens[i];
      iree_tokenizer_token_attr_t attrs = IREE_TOKENIZER_TOKEN_ATTR_NONE;
      if (at->flags & IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL) {
        attrs |= IREE_TOKENIZER_TOKEN_ATTR_SPECIAL;
      }
      iree_string_view_t content =
          iree_tokenizer_huggingface_added_token_content(added_tokens, at);
      status = iree_tokenizer_vocab_builder_ensure_token(builder, at->id,
                                                         content, 0.0f, attrs);
    }
  }

  // Set unk_token special ID on the builder.
  if (iree_status_is_ok(status) &&
      unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    status = iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, (int32_t)unk_token_id);
  }

  // Build final vocab.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(builder, &vocab);
    if (iree_status_is_ok(status)) {
      builder = NULL;  // Consumed by build on success.
    }
  }

  // Allocate WordPiece model.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_wordpiece_model_allocate(
        vocab, iree_make_string_view(prefix_buffer, prefix_length),
        (iree_host_size_t)max_input_chars_per_word,
        IREE_TOKENIZER_WORDPIECE_FLAG_NONE, allocator, &model);
  }

  if (iree_status_is_ok(status)) {
    *out_model = model;
    *out_vocab = vocab;
  } else {
    iree_tokenizer_model_free(model);
    iree_tokenizer_vocab_free(vocab);
    iree_tokenizer_vocab_builder_free(builder);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Unigram Model Parser
//===----------------------------------------------------------------------===//

// Context for Unigram vocab array enumeration.
typedef struct iree_tokenizer_unigram_vocab_parse_context_t {
  iree_tokenizer_vocab_builder_t* builder;
  char unescape_buffer[IREE_TOKENIZER_MAX_MERGE_TOKEN_LENGTH];
  iree_status_t status;
} iree_tokenizer_unigram_vocab_parse_context_t;

// Visitor callback for parsing Unigram vocab array elements.
// Each element is [token_string, score] tuple.
static iree_status_t iree_tokenizer_unigram_vocab_entry_visitor(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  iree_tokenizer_unigram_vocab_parse_context_t* context =
      (iree_tokenizer_unigram_vocab_parse_context_t*)user_data;

  if (type != IREE_JSON_VALUE_TYPE_ARRAY) {
    context->status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Unigram vocab entry at index %" PRIhsz
                         " is not an array (expected [token, score])",
                         index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Get token string (element 0).
  iree_string_view_t token_json = iree_string_view_empty();
  iree_status_t status = iree_json_array_get(value, 0, &token_json);
  if (!iree_status_is_ok(status)) {
    context->status = iree_status_annotate_f(
        status, "Unigram vocab entry at index %" PRIhsz " missing token",
        index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Unescape the token string.
  iree_string_view_t token_text = iree_string_view_empty();
  status = iree_tokenizer_model_json_unescape(
      token_json, context->unescape_buffer, sizeof(context->unescape_buffer),
      &token_text);
  if (!iree_status_is_ok(status)) {
    context->status = iree_status_annotate_f(
        status, "Unigram vocab entry at index %" PRIhsz " unescape failed",
        index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Get score (element 1).
  iree_string_view_t score_json = iree_string_view_empty();
  status = iree_json_array_get(value, 1, &score_json);
  if (!iree_status_is_ok(status)) {
    context->status = iree_status_annotate_f(
        status, "Unigram vocab entry at index %" PRIhsz " missing score",
        index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Parse the score as double (JSON numbers) then convert to float.
  double score_value = 0.0;
  status = iree_json_parse_double(score_json, &score_value);
  if (!iree_status_is_ok(status)) {
    context->status = iree_status_annotate_f(
        status, "Unigram vocab entry at index %" PRIhsz " invalid score",
        index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Add token with index as ID and parsed score.
  status = iree_tokenizer_vocab_builder_add_token_with_id(
      context->builder, (int32_t)index, token_text, (float)score_value,
      IREE_TOKENIZER_TOKEN_ATTR_NONE);
  if (!iree_status_is_ok(status)) {
    context->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  return iree_ok_status();
}

// Parses Unigram vocab array into a vocab builder.
static iree_status_t iree_tokenizer_parse_unigram_vocab_array(
    iree_string_view_t vocab_json, iree_tokenizer_vocab_builder_t* builder,
    iree_allocator_t allocator) {
  (void)allocator;  // Unused after removing dynamic buffer.
  iree_tokenizer_unigram_vocab_parse_context_t context = {
      .builder = builder,
      .unescape_buffer = {0},
      .status = iree_ok_status(),
  };

  iree_status_t status = iree_json_enumerate_array_typed(
      vocab_json, iree_tokenizer_unigram_vocab_entry_visitor, &context);

  if (!iree_status_is_ok(context.status)) {
    if (iree_status_is_ok(status) ||
        iree_status_code(status) == IREE_STATUS_CANCELLED) {
      iree_status_ignore(status);
      return context.status;
    }
    iree_status_ignore(status);
    return context.status;
  }

  if (iree_status_code(status) == IREE_STATUS_CANCELLED) {
    iree_status_ignore(status);
    return iree_ok_status();
  }

  return status;
}

iree_status_t iree_tokenizer_huggingface_parse_unigram_model(
    iree_string_view_t model_json,
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model,
    iree_tokenizer_vocab_t** out_vocab) {
  IREE_ASSERT_ARGUMENT(out_model);
  IREE_ASSERT_ARGUMENT(out_vocab);
  *out_model = NULL;
  *out_vocab = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_validate_object_keys(model_json, kUnigramAllowedKeys,
                                     IREE_ARRAYSIZE(kUnigramAllowedKeys)),
      "in Unigram model object");

  // Look up vocab array.
  iree_string_view_t vocab_json = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_lookup_object_value(model_json, IREE_SV("vocab"), &vocab_json),
      "Unigram model missing vocab");

  // Parse unk_id (integer index into vocab array, or null/-1 for none).
  int64_t unk_id_raw = -1;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_int64(model_json, IREE_SV("unk_id"), -1,
                                     &unk_id_raw));
  iree_tokenizer_token_id_t unk_token_id =
      (unk_id_raw >= 0 && unk_id_raw <= INT32_MAX)
          ? (iree_tokenizer_token_id_t)unk_id_raw
          : IREE_TOKENIZER_TOKEN_ID_INVALID;

  // Parse byte_fallback (default true for SentencePiece compatibility).
  bool byte_fallback = true;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_bool(model_json, IREE_SV("byte_fallback"), true,
                                    &byte_fallback));

  // Resources that need cleanup.
  iree_tokenizer_vocab_builder_t* builder = NULL;
  iree_tokenizer_vocab_t* vocab = NULL;
  iree_tokenizer_model_t* model = NULL;

  // Estimate capacity and create builder.
  iree_host_size_t capacity =
      iree_tokenizer_huggingface_estimate_vocab_capacity(model_json.size);
  iree_status_t status =
      iree_tokenizer_vocab_builder_allocate(capacity, allocator, &builder);

  // Parse vocab array into builder.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_parse_unigram_vocab_array(vocab_json, builder,
                                                      allocator);
  }

  // Process added_tokens.
  if (iree_status_is_ok(status) && added_tokens) {
    for (iree_host_size_t i = 0;
         i < added_tokens->count && iree_status_is_ok(status); ++i) {
      const iree_tokenizer_huggingface_added_token_t* at =
          &added_tokens->tokens[i];
      iree_tokenizer_token_attr_t attrs = IREE_TOKENIZER_TOKEN_ATTR_NONE;
      if (at->flags & IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL) {
        attrs |= IREE_TOKENIZER_TOKEN_ATTR_SPECIAL;
      }
      iree_string_view_t content =
          iree_tokenizer_huggingface_added_token_content(added_tokens, at);
      status = iree_tokenizer_vocab_builder_ensure_token(builder, at->id,
                                                         content, 0.0f, attrs);
    }
  }

  // Set UNK special token if valid.
  if (iree_status_is_ok(status) &&
      unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    status = iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, (int32_t)unk_token_id);
  }

  // Build final vocab.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(builder, &vocab);
    if (iree_status_is_ok(status)) {
      builder = NULL;  // Consumed by build on success.
    }
  }

  // Get UNK score from vocab (for model allocation).
  float unk_score = -10.0f;  // Default if no UNK.
  if (iree_status_is_ok(status) &&
      unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    unk_score = iree_tokenizer_vocab_token_score(vocab, unk_token_id);
  }

  // Map flags.
  iree_tokenizer_unigram_flags_t flags = IREE_TOKENIZER_UNIGRAM_FLAG_NONE;
  if (!byte_fallback) {
    flags |= IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK;
  }

  // Allocate Unigram model.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_unigram_model_allocate(
        vocab, unk_token_id, unk_score, flags, allocator, &model);
  }

  if (iree_status_is_ok(status)) {
    *out_model = model;
    *out_vocab = vocab;
  } else {
    iree_tokenizer_model_free(model);
    iree_tokenizer_vocab_free(vocab);
    iree_tokenizer_vocab_builder_free(builder);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Generic Model Parser
//===----------------------------------------------------------------------===//

// Detects the model type from JSON structure.
static iree_status_t iree_tokenizer_detect_model_type(
    iree_string_view_t model_json, iree_tokenizer_model_type_t* out_type) {
  *out_type = IREE_TOKENIZER_MODEL_TYPE_UNKNOWN;

  // Check for explicit type field.
  iree_string_view_t type_str = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      model_json, IREE_SV("type"), &type_str));

  if (type_str.size > 0) {
    if (iree_string_view_equal(type_str, IREE_SV("BPE"))) {
      *out_type = IREE_TOKENIZER_MODEL_TYPE_BPE;
      return iree_ok_status();
    } else if (iree_string_view_equal(type_str, IREE_SV("WordPiece"))) {
      *out_type = IREE_TOKENIZER_MODEL_TYPE_WORDPIECE;
      return iree_ok_status();
    } else if (iree_string_view_equal(type_str, IREE_SV("Unigram"))) {
      *out_type = IREE_TOKENIZER_MODEL_TYPE_UNIGRAM;
      return iree_ok_status();
    } else {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported model type: '%.*s'",
                              (int)type_str.size, type_str.data);
    }
  }

  // No explicit type - infer from structure.
  // BPE has "merges" array.
  iree_string_view_t merges = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      model_json, IREE_SV("merges"), &merges));
  if (merges.size > 0) {
    *out_type = IREE_TOKENIZER_MODEL_TYPE_BPE;
    return iree_ok_status();
  }

  // WordPiece has "continuing_subword_prefix".
  iree_string_view_t prefix = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      model_json, IREE_SV("continuing_subword_prefix"), &prefix));
  if (prefix.size > 0) {
    *out_type = IREE_TOKENIZER_MODEL_TYPE_WORDPIECE;
    return iree_ok_status();
  }

  // Unigram has "unk_id" as integer.
  iree_string_view_t unk_id = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      model_json, IREE_SV("unk_id"), &unk_id));
  if (unk_id.size > 0) {
    *out_type = IREE_TOKENIZER_MODEL_TYPE_UNIGRAM;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "could not determine model type: no type field and structure does "
      "not match BPE (has merges), WordPiece (has continuing_subword_prefix),"
      " or Unigram (has unk_id)");
}

iree_status_t iree_tokenizer_huggingface_parse_model(
    iree_string_view_t model_json,
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_tokenizer_huggingface_pre_tokenizer_flags_t pre_tokenizer_flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model,
    iree_tokenizer_vocab_t** out_vocab) {
  IREE_ASSERT_ARGUMENT(out_model);
  IREE_ASSERT_ARGUMENT(out_vocab);
  *out_model = NULL;
  *out_vocab = NULL;

  // Auto-detect model type from JSON.
  iree_tokenizer_model_type_t model_type = IREE_TOKENIZER_MODEL_TYPE_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_detect_model_type(model_json, &model_type));

  // Map format-level pre_tokenizer flags to BPE-specific flags.
  iree_tokenizer_bpe_flags_t extra_flags = IREE_TOKENIZER_BPE_FLAG_NONE;
  if (iree_any_bit_set(
          pre_tokenizer_flags,
          IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL)) {
    extra_flags |= IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT;
  }
  if (iree_any_bit_set(
          pre_tokenizer_flags,
          IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT)) {
    extra_flags |= IREE_TOKENIZER_BPE_FLAG_ENABLE_WORD_CACHE;
  }

  // Dispatch to type-specific parser.
  switch (model_type) {
    case IREE_TOKENIZER_MODEL_TYPE_BPE:
      return iree_tokenizer_huggingface_parse_bpe_model(
          model_json, added_tokens, extra_flags, allocator, out_model,
          out_vocab);

    case IREE_TOKENIZER_MODEL_TYPE_WORDPIECE:
      return iree_tokenizer_huggingface_parse_wordpiece_model(
          model_json, added_tokens, allocator, out_model, out_vocab);

    case IREE_TOKENIZER_MODEL_TYPE_UNIGRAM:
      return iree_tokenizer_huggingface_parse_unigram_model(
          model_json, added_tokens, allocator, out_model, out_vocab);

    case IREE_TOKENIZER_MODEL_TYPE_UNKNOWN:
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown model type");
  }
}
