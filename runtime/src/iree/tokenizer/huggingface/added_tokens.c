// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/added_tokens.h"

#include "iree/base/internal/json.h"
#include "iree/tokenizer/literals.h"

//===----------------------------------------------------------------------===//
// Added Tokens Storage
//===----------------------------------------------------------------------===//

void iree_tokenizer_added_tokens_initialize(
    iree_tokenizer_added_tokens_t* added_tokens, iree_allocator_t allocator) {
  memset(added_tokens, 0, sizeof(*added_tokens));
  added_tokens->allocator = allocator;
}

void iree_tokenizer_added_tokens_deinitialize(
    iree_tokenizer_added_tokens_t* added_tokens) {
  for (iree_host_size_t i = 0; i < added_tokens->count; ++i) {
    if (added_tokens->entries[i].content) {
      iree_allocator_free(added_tokens->allocator,
                          added_tokens->entries[i].content);
    }
  }
  if (added_tokens->entries) {
    iree_allocator_free(added_tokens->allocator, added_tokens->entries);
  }
  memset(added_tokens, 0, sizeof(*added_tokens));
}

iree_status_t iree_tokenizer_added_tokens_add(
    iree_tokenizer_added_tokens_t* added_tokens, int32_t id,
    iree_string_view_t content, bool is_special,
    iree_tokenizer_special_token_t special_type) {
  // Skip duplicates by ID (first entry wins).
  for (iree_host_size_t i = 0; i < added_tokens->count; ++i) {
    if (added_tokens->entries[i].id == id) {
      return iree_ok_status();
    }
  }

  // Grow if needed.
  if (added_tokens->count >= added_tokens->capacity) {
    iree_host_size_t new_capacity = added_tokens->capacity * 2;
    if (new_capacity < 16) new_capacity = 16;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        added_tokens->allocator,
        new_capacity * sizeof(iree_tokenizer_added_token_entry_t),
        (void**)&added_tokens->entries));
    added_tokens->capacity = new_capacity;
  }

  // Copy the content string.
  char* content_copy = NULL;
  if (content.size > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        added_tokens->allocator, content.size, (void**)&content_copy));
    memcpy(content_copy, content.data, content.size);
  }

  added_tokens->entries[added_tokens->count] =
      (iree_tokenizer_added_token_entry_t){
          .id = id,
          .content = content_copy,
          .content_length = content.size,
          .is_special = is_special,
          .found_in_vocab = false,
          .special_type = special_type,
      };
  added_tokens->count++;
  return iree_ok_status();
}

// O(n) linear search is acceptable: only called during tokenizer loading (not
// on the hot path), and added_tokens count is typically small (<20 entries).
iree_tokenizer_added_token_entry_t* iree_tokenizer_added_tokens_find(
    iree_tokenizer_added_tokens_t* added_tokens, int32_t id) {
  for (iree_host_size_t i = 0; i < added_tokens->count; ++i) {
    if (added_tokens->entries[i].id == id) {
      return &added_tokens->entries[i];
    }
  }
  return NULL;
}

// Maximum token text size for special token detection.
// Real special tokens are tiny (e.g., "[UNK]", "<|endoftext|>").
#define IREE_TOKENIZER_MAX_SPECIAL_TOKEN_SIZE 64

// Context for fallback special token scanning.
typedef struct {
  iree_tokenizer_vocab_builder_t* builder;
  iree_string_view_t unk_token;  // From model.unk_token if present.
  char unescape_buffer[IREE_TOKENIZER_MAX_SPECIAL_TOKEN_SIZE];
  iree_status_t status;
} iree_tokenizer_fallback_special_ctx_t;

static iree_status_t iree_tokenizer_fallback_special_visitor(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  iree_tokenizer_fallback_special_ctx_t* ctx =
      (iree_tokenizer_fallback_special_ctx_t*)user_data;

  // Parse the token ID.
  int64_t token_id;
  iree_status_t status = iree_json_parse_int64(value, &token_id);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (token_id < 0 || token_id > INT32_MAX) {
    return iree_ok_status();  // Skip out of range.
  }

  // Unescape the key into fixed buffer.
  iree_host_size_t unescaped_length;
  status = iree_json_unescape_string(key, 0, NULL, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (unescaped_length > IREE_TOKENIZER_MAX_SPECIAL_TOKEN_SIZE) {
    ctx->status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "token text exceeds maximum size of %d for special token detection",
        IREE_TOKENIZER_MAX_SPECIAL_TOKEN_SIZE);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  status = iree_json_unescape_string(key, IREE_TOKENIZER_MAX_SPECIAL_TOKEN_SIZE,
                                     ctx->unescape_buffer, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  iree_string_view_t text =
      iree_make_string_view(ctx->unescape_buffer, unescaped_length);

  // Check if this matches model.unk_token.
  if (ctx->unk_token.size > 0 && iree_string_view_equal(text, ctx->unk_token)) {
    iree_tokenizer_vocab_builder_set_special_token(
        ctx->builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, (int32_t)token_id);
  }

  // Check for known special token patterns.
  iree_tokenizer_special_token_t special_type =
      iree_tokenizer_match_special_token(text);
  if ((int)special_type >= 0) {
    iree_tokenizer_vocab_builder_set_special_token(ctx->builder, special_type,
                                                   (int32_t)token_id);
    // Also set ATTR_SPECIAL so skip_special_tokens works during decode.
    status = iree_tokenizer_vocab_builder_add_token_attrs(
        ctx->builder, (int32_t)token_id, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
    if (!iree_status_is_ok(status)) {
      ctx->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
  }

  return iree_ok_status();
}

iree_status_t iree_tokenizer_detect_specials_from_vocab(
    iree_string_view_t model, iree_tokenizer_vocab_builder_t* builder) {
  iree_status_t status = iree_ok_status();

  // Get and unescape model.unk_token into a stack buffer.
  char unk_token_buffer[IREE_TOKENIZER_MAX_SPECIAL_TOKEN_SIZE];
  iree_string_view_t unk_token = iree_string_view_empty();
  iree_string_view_t unk_token_json = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = iree_json_try_lookup_object_value(model, IREE_SV("unk_token"),
                                               &unk_token_json);
  }
  if (iree_status_is_ok(status) && unk_token_json.size > 0) {
    iree_host_size_t unk_length = 0;
    status = iree_json_unescape_string(unk_token_json, 0, NULL, &unk_length);
    if (iree_status_is_ok(status) && unk_length > 0) {
      if (unk_length > IREE_TOKENIZER_MAX_SPECIAL_TOKEN_SIZE) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "unk_token exceeds maximum size of %d",
                                  IREE_TOKENIZER_MAX_SPECIAL_TOKEN_SIZE);
      }
    }
    if (iree_status_is_ok(status) && unk_length > 0) {
      status = iree_json_unescape_string(unk_token_json, unk_length,
                                         unk_token_buffer, &unk_length);
      if (iree_status_is_ok(status)) {
        unk_token = iree_make_string_view(unk_token_buffer, unk_length);
      }
    }
  }

  // Get vocab.
  iree_string_view_t vocab = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = iree_json_try_lookup_object_value(model, IREE_SV("vocab"), &vocab);
  }
  if (!iree_status_is_ok(status) || vocab.size == 0) {
    return status;
  }

  // Scan vocab for special tokens.
  iree_tokenizer_fallback_special_ctx_t ctx = {
      .builder = builder,
      .unk_token = unk_token,
      .status = iree_ok_status(),
  };
  status = iree_json_enumerate_object(
      vocab, iree_tokenizer_fallback_special_visitor, &ctx);

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
// JSON Parsing
//===----------------------------------------------------------------------===//

typedef struct {
  iree_tokenizer_added_tokens_t* added_tokens;
  iree_allocator_t allocator;
  char* unescape_buffer;
  iree_host_size_t unescape_capacity;
  iree_status_t status;
} iree_tokenizer_parse_added_ctx_t;

static iree_status_t iree_tokenizer_parse_added_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t value) {
  (void)index;
  iree_tokenizer_parse_added_ctx_t* ctx =
      (iree_tokenizer_parse_added_ctx_t*)user_data;

  // Get the token ID.
  iree_string_view_t id_value;
  iree_status_t status =
      iree_json_lookup_object_value(value, IREE_SV("id"), &id_value);
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

  // Check if this entry has "special": true (optional, defaults to false).
  iree_string_view_t special_value;
  status = iree_json_try_lookup_object_value(value, IREE_SV("special"),
                                             &special_value);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  bool is_special = iree_string_view_equal(special_value, IREE_SV("true"));

  // Get the content string.
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
    iree_host_size_t new_capacity = ctx->unescape_capacity * 2;
    if (new_capacity < unescaped_length) new_capacity = unescaped_length;
    if (new_capacity < 64) new_capacity = 64;
    status = iree_allocator_realloc(ctx->allocator, new_capacity,
                                    (void**)&ctx->unescape_buffer);
    if (!iree_status_is_ok(status)) {
      ctx->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
    ctx->unescape_capacity = new_capacity;
  }

  status = iree_json_unescape_string(content_value, ctx->unescape_capacity,
                                     ctx->unescape_buffer, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  iree_string_view_t content =
      iree_make_string_view(ctx->unescape_buffer, unescaped_length);

  // Match against known special token patterns.
  iree_tokenizer_special_token_t special_type =
      is_special ? iree_tokenizer_match_special_token(content)
                 : (iree_tokenizer_special_token_t)-1;

  // Add to storage.
  status = iree_tokenizer_added_tokens_add(ctx->added_tokens, (int32_t)token_id,
                                           content, is_special, special_type);
  if (!iree_status_is_ok(status)) {
    ctx->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  return iree_ok_status();
}

iree_status_t iree_tokenizer_added_tokens_parse_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_added_tokens_t* added_tokens) {
  iree_string_view_t added_tokens_json;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      json_root, IREE_SV("added_tokens"), &added_tokens_json));
  if (added_tokens_json.size == 0) {
    return iree_ok_status();  // No added_tokens, that's fine.
  }

  iree_tokenizer_parse_added_ctx_t ctx = {
      .added_tokens = added_tokens,
      .allocator = allocator,
      .unescape_buffer = NULL,
      .unescape_capacity = 0,
      .status = iree_ok_status(),
  };

  iree_status_t status = iree_json_enumerate_array(
      added_tokens_json, iree_tokenizer_parse_added_visitor, &ctx);

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
// Finalization
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_added_tokens_finalize(
    iree_tokenizer_added_tokens_t* added_tokens,
    iree_tokenizer_vocab_builder_t* builder) {
  for (iree_host_size_t i = 0; i < added_tokens->count; ++i) {
    iree_tokenizer_added_token_entry_t* entry = &added_tokens->entries[i];

    // Add token if it wasn't in model.vocab.
    if (!entry->found_in_vocab) {
      iree_tokenizer_token_attr_t attrs = IREE_TOKENIZER_TOKEN_ATTR_NONE;
      if (entry->is_special) {
        attrs |= IREE_TOKENIZER_TOKEN_ATTR_SPECIAL;
      }
      iree_string_view_t content =
          iree_make_string_view(entry->content, entry->content_length);
      IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_add_token_with_id(
          builder, entry->id, content, 0.0f, attrs));
    }

    // Set special token ID mapping if this is a recognized special token.
    if ((int)entry->special_type >= 0) {
      IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_set_special_token(
          builder, entry->special_type, entry->id));
    }
  }
  return iree_ok_status();
}
