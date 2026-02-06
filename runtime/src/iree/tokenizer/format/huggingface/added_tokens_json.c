// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/added_tokens_json.h"

#include "iree/base/internal/json.h"

//===----------------------------------------------------------------------===//
// Added Token Parsing
//===----------------------------------------------------------------------===//

// Allowed keys in an added token object.
// Reference: tokenizers/src/tokenizer/added_vocabulary.rs AddedToken struct
static const iree_string_view_t kAddedTokenAllowedKeys[] = {
    IREE_SVL("id"),      IREE_SVL("content"), IREE_SVL("single_word"),
    IREE_SVL("lstrip"),  IREE_SVL("rstrip"),  IREE_SVL("normalized"),
    IREE_SVL("special"),
};

// Context for the first pass: counting tokens and total content size.
typedef struct {
  iree_host_size_t token_count;
  iree_host_size_t total_content_size;
  iree_status_t status;
} iree_tokenizer_huggingface_added_tokens_count_ctx_t;

// Visitor for counting pass.
static iree_status_t iree_tokenizer_huggingface_added_tokens_count_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t token_json) {
  iree_tokenizer_huggingface_added_tokens_count_ctx_t* ctx =
      (iree_tokenizer_huggingface_added_tokens_count_ctx_t*)user_data;

  // Validate keys.
  iree_status_t status =
      iree_json_validate_object_keys(token_json, kAddedTokenAllowedKeys,
                                     IREE_ARRAYSIZE(kAddedTokenAllowedKeys));
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Look up required content field to get its size.
  iree_string_view_t content_value = iree_string_view_empty();
  status = iree_json_lookup_object_value(token_json, IREE_SV("content"),
                                         &content_value);
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Query unescaped length (out_string=NULL computes length without writing).
  iree_host_size_t unescaped_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(content_value, 0, NULL, &unescaped_length));

  // Overflow-safe accumulation of token count.
  iree_host_size_t new_token_count = 0;
  if (!iree_host_size_checked_add(ctx->token_count, 1, &new_token_count)) {
    ctx->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "added_tokens count overflow");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  ctx->token_count = new_token_count;

  // Overflow-safe accumulation of content size (+1 for NUL terminator).
  iree_host_size_t content_with_nul = 0;
  if (!iree_host_size_checked_add(unescaped_length, 1, &content_with_nul)) {
    ctx->status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "added_tokens[%" PRIhsz "].content length overflow", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  iree_host_size_t new_total = 0;
  if (!iree_host_size_checked_add(ctx->total_content_size, content_with_nul,
                                  &new_total)) {
    ctx->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "added_tokens total content size overflow");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  ctx->total_content_size = new_total;

  return iree_ok_status();
}

// Context for the second pass: parsing tokens into the allocated buffer.
typedef struct {
  // Total allocated sizes for bounds checking.
  iree_host_size_t tokens_array_size;
  iree_host_size_t string_pool_size;
  // Current write positions.
  iree_host_size_t current_token_index;
  iree_host_size_t string_pool_offset;
  // Output pointers.
  iree_tokenizer_huggingface_added_token_t* tokens;
  char* string_pool;
  // Error status.
  iree_status_t status;
} iree_tokenizer_huggingface_added_tokens_parse_ctx_t;

// Visitor for parsing pass.
static iree_status_t iree_tokenizer_huggingface_added_tokens_parse_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t token_json) {
  iree_tokenizer_huggingface_added_tokens_parse_ctx_t* ctx =
      (iree_tokenizer_huggingface_added_tokens_parse_ctx_t*)user_data;

  // Bounds check: ensure we have room for this token.
  if (ctx->current_token_index *
          sizeof(iree_tokenizer_huggingface_added_token_t) >=
      ctx->tokens_array_size) {
    ctx->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "token index exceeds allocated array");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  iree_tokenizer_huggingface_added_token_t* token =
      &ctx->tokens[ctx->current_token_index];

  // Parse required id field.
  iree_string_view_t id_value = iree_string_view_empty();
  iree_status_t status =
      iree_json_lookup_object_value(token_json, IREE_SV("id"), &id_value);
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  int64_t id_int = 0;
  status = iree_json_parse_int64(id_value, &id_int);
  if (!iree_status_is_ok(status)) {
    ctx->status = iree_status_annotate_f(
        status, "in added_tokens[%" PRIhsz "].id", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (id_int < 0 || id_int > INT32_MAX) {
    ctx->status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                   "added_tokens[%" PRIhsz
                                   "].id must be non-negative: %" PRId64,
                                   index, id_int);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  token->id = (iree_tokenizer_token_id_t)id_int;

  // Parse required boolean flags.
  // Reference: tokenizers/src/tokenizer/added_vocabulary.rs AddedToken struct
  // has no #[serde(default)] on any field — all are required in the JSON.
  bool single_word = false;
  status =
      iree_json_lookup_bool(token_json, IREE_SV("single_word"), &single_word);
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  bool lstrip = false;
  status = iree_json_lookup_bool(token_json, IREE_SV("lstrip"), &lstrip);
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  bool rstrip = false;
  status = iree_json_lookup_bool(token_json, IREE_SV("rstrip"), &rstrip);
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  bool special = false;
  status = iree_json_lookup_bool(token_json, IREE_SV("special"), &special);
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  bool normalized = false;
  status =
      iree_json_lookup_bool(token_json, IREE_SV("normalized"), &normalized);
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Build flags.
  iree_tokenizer_huggingface_added_token_flags_t flags =
      IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_NONE;
  if (single_word)
    flags |= IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SINGLE_WORD;
  if (lstrip) flags |= IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_LSTRIP;
  if (rstrip) flags |= IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_RSTRIP;
  if (normalized)
    flags |= IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_NORMALIZED;
  if (special) flags |= IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL;
  token->flags = flags;

  // Parse required content field.
  iree_string_view_t content_value = iree_string_view_empty();
  status = iree_json_lookup_object_value(token_json, IREE_SV("content"),
                                         &content_value);
  if (!iree_status_is_ok(status)) {
    ctx->status =
        iree_status_annotate_f(status, "in added_tokens[%" PRIhsz "]", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Calculate remaining space in string pool (bounds check).
  if (ctx->string_pool_offset > ctx->string_pool_size) {
    ctx->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "string pool offset exceeds allocation");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  iree_host_size_t remaining_size =
      ctx->string_pool_size - ctx->string_pool_offset;

  // Record content offset before writing.
  if (ctx->string_pool_offset > UINT32_MAX) {
    ctx->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "string pool offset exceeds uint32_t");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  token->content_offset = (uint32_t)ctx->string_pool_offset;

  // Unescape content into the string pool.
  char* content_ptr = ctx->string_pool + ctx->string_pool_offset;
  iree_host_size_t unescaped_length = 0;
  status = iree_json_unescape_string(content_value, remaining_size, content_ptr,
                                     &unescaped_length);
  if (!iree_status_is_ok(status)) {
    ctx->status = iree_status_annotate_f(
        status, "in added_tokens[%" PRIhsz "].content", index);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Validate content length fits in uint32_t.
  if (unescaped_length > UINT32_MAX) {
    ctx->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "content length exceeds uint32_t");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  token->content_length = (uint32_t)unescaped_length;

  // NUL-terminate for debugging convenience.
  // Bounds check: need room for content + NUL.
  iree_host_size_t content_with_nul = 0;
  if (!iree_host_size_checked_add(unescaped_length, 1, &content_with_nul) ||
      content_with_nul > remaining_size) {
    ctx->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "content + NUL exceeds remaining space");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  content_ptr[unescaped_length] = '\0';

  // Advance string pool offset with overflow check.
  iree_host_size_t new_offset = 0;
  if (!iree_host_size_checked_add(ctx->string_pool_offset, content_with_nul,
                                  &new_offset)) {
    ctx->status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "string pool offset overflow");
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  ctx->string_pool_offset = new_offset;
  ctx->current_token_index++;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifetime
//===----------------------------------------------------------------------===//

void iree_tokenizer_huggingface_added_tokens_free(
    iree_tokenizer_huggingface_added_tokens_t* tokens) {
  if (!tokens || tokens->count == 0) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = tokens->allocator;
  iree_allocator_free(allocator, (void*)tokens->tokens);
  memset(tokens, 0, sizeof(*tokens));
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_huggingface_parse_added_tokens_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_huggingface_added_tokens_t* out_tokens) {
  IREE_ASSERT_ARGUMENT(out_tokens);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_tokens, 0, sizeof(*out_tokens));

  // Look up the added_tokens field (optional).
  iree_string_view_t added_tokens_array = iree_string_view_empty();
  iree_status_t status = iree_json_try_lookup_object_value(
      json_root, IREE_SV("added_tokens"), &added_tokens_array);

  // If no added_tokens or null, return empty list.
  if (iree_status_is_ok(status) &&
      (added_tokens_array.size == 0 ||
       iree_string_view_equal(added_tokens_array, IREE_SV("null")))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // First pass: count tokens and total content size (with overflow checks).
  iree_tokenizer_huggingface_added_tokens_count_ctx_t count_ctx = {
      .token_count = 0,
      .total_content_size = 0,
      .status = iree_ok_status(),
  };
  if (iree_status_is_ok(status)) {
    status = iree_json_enumerate_array(
        added_tokens_array,
        iree_tokenizer_huggingface_added_tokens_count_visitor, &count_ctx);
    // Check for visitor errors first. CANCELLED returns from the visitor are
    // swallowed by enumerate_array (treated as early-exit), so we must check
    // the context's saved status.
    if (!iree_status_is_ok(count_ctx.status)) {
      iree_status_ignore(status);
      status = count_ctx.status;
    }
  }

  // Empty array is valid.
  if (iree_status_is_ok(status) && count_ctx.token_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Calculate allocation size using checked math.
  // Layout: [tokens array (aligned)][string pool]
  iree_host_size_t tokens_array_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(
          count_ctx.token_count,
          sizeof(iree_tokenizer_huggingface_added_token_t),
          &tokens_array_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "added_tokens array size overflow");
  }

  // Validate total sizes fit in uint32_t (for offsets/lengths).
  if (iree_status_is_ok(status) && tokens_array_size > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "added_tokens array exceeds 4GB limit");
  }
  if (iree_status_is_ok(status) && count_ctx.total_content_size > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "added_tokens string pool exceeds 4GB limit");
  }

  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_add(tokens_array_size,
                                  count_ctx.total_content_size, &total_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "added_tokens total size overflow");
  }

  // Allocate combined buffer.
  uint8_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, total_size, (void**)&buffer);
  }

  // Second pass: parse tokens into the buffer.
  iree_tokenizer_huggingface_added_token_t* tokens_array = NULL;
  char* string_pool = NULL;
  iree_tokenizer_huggingface_added_tokens_parse_ctx_t parse_ctx = {0};
  if (iree_status_is_ok(status)) {
    // The token array is at the start (naturally aligned since buffer is
    // aligned).
    tokens_array = (iree_tokenizer_huggingface_added_token_t*)buffer;
    string_pool = (char*)(buffer + tokens_array_size);

    parse_ctx = (iree_tokenizer_huggingface_added_tokens_parse_ctx_t){
        .tokens_array_size = tokens_array_size,
        .string_pool_size = count_ctx.total_content_size,
        .current_token_index = 0,
        .string_pool_offset = 0,
        .tokens = tokens_array,
        .string_pool = string_pool,
        .status = iree_ok_status(),
    };
    status = iree_json_enumerate_array(
        added_tokens_array,
        iree_tokenizer_huggingface_added_tokens_parse_visitor, &parse_ctx);
    // Check for visitor errors first (CANCELLED is swallowed by
    // enumerate_array).
    if (!iree_status_is_ok(parse_ctx.status)) {
      iree_status_ignore(status);
      status = parse_ctx.status;
    }
  }

  // Sanity check: verify we used exactly the space we allocated.
  if (iree_status_is_ok(status) &&
      parse_ctx.current_token_index != count_ctx.token_count) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "token count mismatch: expected %" PRIhsz ", got %" PRIhsz,
        count_ctx.token_count, parse_ctx.current_token_index);
  }
  if (iree_status_is_ok(status) &&
      parse_ctx.string_pool_offset != count_ctx.total_content_size) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "string pool size mismatch: expected %" PRIhsz ", got %" PRIhsz,
        count_ctx.total_content_size, parse_ctx.string_pool_offset);
  }

  if (iree_status_is_ok(status)) {
    // Success - fill output struct.
    out_tokens->count = count_ctx.token_count;
    out_tokens->string_pool_size = count_ctx.total_content_size;
    out_tokens->allocator = allocator;
    out_tokens->tokens = tokens_array;
    out_tokens->string_pool = string_pool;
  } else {
    iree_allocator_free(allocator, buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
