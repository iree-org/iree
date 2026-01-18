// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/literals_json.h"

#include "iree/base/internal/json.h"

//===----------------------------------------------------------------------===//
// JSON Parsing Context
//===----------------------------------------------------------------------===//

typedef struct {
  iree_tokenizer_literals_t* literals;
  char* unescape_buffer;
  iree_host_size_t unescape_capacity;
  iree_status_t status;
} iree_tokenizer_literals_parse_ctx_t;

//===----------------------------------------------------------------------===//
// JSON Parsing
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_literals_parse_entry(
    void* user_data, iree_host_size_t index, iree_string_view_t value) {
  (void)index;
  iree_tokenizer_literals_parse_ctx_t* ctx =
      (iree_tokenizer_literals_parse_ctx_t*)user_data;

  // Validate allowed keys for added_token entry.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("id"),         IREE_SVL("content"), IREE_SVL("special"),
      IREE_SVL("lstrip"),     IREE_SVL("rstrip"),  IREE_SVL("single_word"),
      IREE_SVL("normalized"),
  };
  iree_status_t status = iree_json_validate_object_keys(
      value, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys));
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Get the token ID (required).
  iree_string_view_t id_value;
  status = iree_json_lookup_object_value(value, IREE_SV("id"), &id_value);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  int64_t token_id;
  status = iree_json_parse_int64(id_value, &token_id);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (token_id < 0 || token_id > INT32_MAX) {
    return iree_ok_status();  // Skip out-of-range IDs.
  }

  // Get the content string (required).
  iree_string_view_t content_value;
  status =
      iree_json_lookup_object_value(value, IREE_SV("content"), &content_value);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (content_value.size == 0) {
    return iree_ok_status();  // Skip empty content.
  }

  // Unescape the content.
  iree_host_size_t unescaped_length;
  status = iree_json_unescape_string(content_value, 0, NULL, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  if (unescaped_length > ctx->unescape_capacity) {
    status = iree_allocator_grow_array(
        ctx->literals->allocator, iree_max(64, unescaped_length),
        /*element_size=*/1, &ctx->unescape_capacity,
        (void**)&ctx->unescape_buffer);
    if (!iree_status_is_ok(status)) {
      ctx->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
  }

  status = iree_json_unescape_string(content_value, ctx->unescape_capacity,
                                     ctx->unescape_buffer, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  iree_string_view_t content =
      iree_make_string_view(ctx->unescape_buffer, unescaped_length);

  // Check if this entry has "special": true (optional, defaults to false).
  iree_string_view_t special_value;
  status = iree_json_try_lookup_object_value(value, IREE_SV("special"),
                                             &special_value);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  bool is_special = iree_string_view_equal(special_value, IREE_SV("true"));

  // Parse flags (all optional, default to false).
  iree_tokenizer_literal_flags_t flags = IREE_TOKENIZER_LITERAL_FLAG_NONE;

  iree_string_view_t flag_value;
  status =
      iree_json_try_lookup_object_value(value, IREE_SV("lstrip"), &flag_value);
  if (iree_status_is_ok(status) &&
      iree_string_view_equal(flag_value, IREE_SV("true"))) {
    flags |= IREE_TOKENIZER_LITERAL_FLAG_LSTRIP;
  }

  status =
      iree_json_try_lookup_object_value(value, IREE_SV("rstrip"), &flag_value);
  if (iree_status_is_ok(status) &&
      iree_string_view_equal(flag_value, IREE_SV("true"))) {
    flags |= IREE_TOKENIZER_LITERAL_FLAG_RSTRIP;
  }

  status = iree_json_try_lookup_object_value(value, IREE_SV("single_word"),
                                             &flag_value);
  if (iree_status_is_ok(status) &&
      iree_string_view_equal(flag_value, IREE_SV("true"))) {
    flags |= IREE_TOKENIZER_LITERAL_FLAG_SINGLE_WORD;
  }

  status = iree_json_try_lookup_object_value(value, IREE_SV("normalized"),
                                             &flag_value);
  if (iree_status_is_ok(status) &&
      iree_string_view_equal(flag_value, IREE_SV("true"))) {
    flags |= IREE_TOKENIZER_LITERAL_FLAG_NORMALIZED;
  }

  if (is_special) {
    flags |= IREE_TOKENIZER_LITERAL_FLAG_SPECIAL;
  }

  // Match against known special token patterns.
  iree_tokenizer_special_token_t special_type =
      is_special ? iree_tokenizer_match_special_token(content)
                 : (iree_tokenizer_special_token_t)-1;

  // Add to literals collection.
  status = iree_tokenizer_literals_add(ctx->literals, (int32_t)token_id,
                                       content, flags, special_type);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  return iree_ok_status();
}

iree_status_t iree_tokenizer_literals_parse_json(
    iree_string_view_t json_root, iree_tokenizer_literals_t* literals) {
  // Look up the added_tokens array.
  iree_string_view_t added_tokens_json;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json_root, IREE_SV("added_tokens"), &added_tokens_json));
  if (added_tokens_json.size == 0) {
    return iree_ok_status();  // No added_tokens, that's fine.
  }

  iree_tokenizer_literals_parse_ctx_t ctx = {
      .literals = literals,
      .unescape_buffer = NULL,
      .unescape_capacity = 0,
      .status = iree_ok_status(),
  };

  iree_status_t status = iree_json_enumerate_array(
      added_tokens_json, iree_tokenizer_literals_parse_entry, &ctx);

  if (ctx.unescape_buffer) {
    iree_allocator_free(literals->allocator, ctx.unescape_buffer);
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
