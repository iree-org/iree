// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/vocab_json.h"

#include "iree/base/internal/json.h"

//===----------------------------------------------------------------------===//
// Model Extraction
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_json_extract_model(iree_string_view_t root,
                                                iree_string_view_t* out_model) {
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(root, IREE_SV("model"), out_model));
  if (out_model->size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tokenizer.json missing 'model' field");
  }
  return iree_ok_status();
}

iree_status_t iree_tokenizer_json_validate_model_type(
    iree_string_view_t model, iree_string_view_t expected_type) {
  iree_string_view_t type_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_object_value(model, IREE_SV("type"), &type_value));

  // Accept missing type (inference already happened) or exact match.
  if (type_value.size == 0 ||
      iree_string_view_equal(type_value, expected_type)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "model.type is not '%.*s'", (int)expected_type.size,
                          expected_type.data);
}

//===----------------------------------------------------------------------===//
// Vocab Counting
//===----------------------------------------------------------------------===//

typedef struct {
  iree_host_size_t count;
} iree_tokenizer_count_vocab_context_t;

static iree_status_t iree_tokenizer_count_vocab_visitor(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  (void)key;
  (void)value;
  iree_tokenizer_count_vocab_context_t* context =
      (iree_tokenizer_count_vocab_context_t*)user_data;
  context->count++;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_json_count_vocab(iree_string_view_t model,
                                              iree_host_size_t* out_count) {
  iree_string_view_t vocab;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(model, IREE_SV("vocab"), &vocab));
  if (vocab.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model missing 'vocab' field");
  }
  iree_tokenizer_count_vocab_context_t context = {.count = 0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object(
      vocab, iree_tokenizer_count_vocab_visitor, &context));
  *out_count = context.count;
  return iree_ok_status();
}

iree_host_size_t iree_tokenizer_json_estimate_vocab_capacity(
    iree_host_size_t json_size) {
  // Typical vocab entry is ~20-30 bytes: "token": 12345,
  // Use conservative estimate of 40 bytes/entry to avoid over-allocation.
  iree_host_size_t estimate = json_size / 40;
  return estimate > 1000 ? estimate : 1000;
}

//===----------------------------------------------------------------------===//
// Vocab Parsing
//===----------------------------------------------------------------------===//

typedef struct {
  iree_tokenizer_vocab_builder_t* builder;
  iree_tokenizer_added_tokens_t* added_tokens;
  iree_allocator_t allocator;
  char* unescape_buffer;
  iree_host_size_t unescape_capacity;
  iree_status_t status;
} iree_tokenizer_parse_vocab_context_t;

static iree_status_t iree_tokenizer_parse_vocab_visitor(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  iree_tokenizer_parse_vocab_context_t* context =
      (iree_tokenizer_parse_vocab_context_t*)user_data;

  // Parse the token ID from the value.
  int64_t token_id;
  iree_status_t status = iree_json_parse_int64(value, &token_id);
  if (!iree_status_is_ok(status)) {
    context->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (token_id < 0 || token_id > INT32_MAX) {
    context->status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE, "token ID out of range: %" PRId64, token_id);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Unescape the key (token text) so we can validate against added_tokens.
  // Since unescaped_length <= escaped_length (escapes like \n shrink), we can
  // ensure the buffer fits the escaped size and unescape directly in one pass.
  if (key.size > context->unescape_capacity) {
    iree_host_size_t new_capacity = context->unescape_capacity * 2;
    if (new_capacity < key.size) new_capacity = key.size;
    if (new_capacity < 256) new_capacity = 256;
    status = iree_allocator_realloc(context->allocator, new_capacity,
                                    (void**)&context->unescape_buffer);
    if (!iree_status_is_ok(status)) {
      context->status = status;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
    context->unescape_capacity = new_capacity;
  }

  iree_host_size_t unescaped_length;
  status =
      iree_json_unescape_string(key, context->unescape_capacity,
                                context->unescape_buffer, &unescaped_length);
  if (!iree_status_is_ok(status)) {
    context->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  iree_string_view_t text =
      iree_make_string_view(context->unescape_buffer, unescaped_length);

  // Check if this token is in added_tokens (to get ATTR_SPECIAL).
  // Also validate that the content matches.
  iree_tokenizer_token_attr_t attrs = IREE_TOKENIZER_TOKEN_ATTR_NONE;
  if (context->added_tokens) {
    iree_tokenizer_added_token_entry_t* added_entry =
        iree_tokenizer_added_tokens_find(context->added_tokens,
                                         (int32_t)token_id);
    if (added_entry) {
      // Validate that added_tokens content matches vocab entry.
      iree_string_view_t added_content = iree_make_string_view(
          added_entry->content, added_entry->content_length);
      if (!iree_string_view_equal(text, added_content)) {
        context->status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "added_tokens content mismatch for ID %" PRId32
                             ": vocab has '%.*s' but added_tokens has '%.*s'",
                             (int32_t)token_id, (int)text.size, text.data,
                             (int)added_content.size, added_content.data);
        return iree_status_from_code(IREE_STATUS_CANCELLED);
      }
      added_entry->found_in_vocab = true;
      if (added_entry->is_special) {
        attrs |= IREE_TOKENIZER_TOKEN_ATTR_SPECIAL;
      }
    }
  }

  // Add token with explicit ID and attrs.
  status = iree_tokenizer_vocab_builder_add_token_with_id(
      context->builder, (int32_t)token_id, text, 0.0f, attrs);
  if (!iree_status_is_ok(status)) {
    context->status = status;
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  return iree_ok_status();
}

iree_status_t iree_tokenizer_json_parse_vocab(
    iree_string_view_t model, iree_allocator_t allocator,
    iree_tokenizer_added_tokens_t* added_tokens,
    iree_tokenizer_vocab_builder_t* builder) {
  iree_string_view_t vocab;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(model, IREE_SV("vocab"), &vocab));

  iree_tokenizer_parse_vocab_context_t context = {
      .builder = builder,
      .added_tokens = added_tokens,
      .allocator = allocator,
      .unescape_buffer = NULL,
      .unescape_capacity = 0,
      .status = iree_ok_status(),
  };

  iree_status_t status = iree_json_enumerate_object(
      vocab, iree_tokenizer_parse_vocab_visitor, &context);

  if (context.unescape_buffer) {
    iree_allocator_free(allocator, context.unescape_buffer);
  }

  if (!iree_status_is_ok(context.status)) {
    iree_status_ignore(status);
    return context.status;
  }
  if (iree_status_is_cancelled(status)) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  return status;
}
