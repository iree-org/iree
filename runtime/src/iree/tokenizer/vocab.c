// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab.h"

#include "iree/base/api.h"
#include "iree/tokenizer/vocab_hash.h"
#include "iree/tokenizer/vocab_internal.h"

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT int32_t iree_tokenizer_vocab_lookup(
    const iree_tokenizer_vocab_t* vocab, iree_string_view_t text) {
  if (!vocab || !vocab->hash) return -1;
  return iree_tokenizer_vocab_hash_lookup(vocab->hash, text);
}

IREE_API_EXPORT iree_string_view_t iree_tokenizer_vocab_token_text(
    const iree_tokenizer_vocab_t* vocab, int32_t token_id) {
  if (!vocab || token_id < 0 || token_id > vocab->max_token_id) {
    return iree_string_view_empty();
  }
  const iree_tokenizer_token_t* token = &vocab->tokens[token_id];
  // Return empty for gap slots (sparse vocab with missing IDs).
  if (iree_any_bit_set(token->attributes, IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
    return iree_string_view_empty();
  }
  return iree_make_string_view(
      (const char*)vocab->string_data + token->string_offset,
      token->string_length);
}

IREE_API_EXPORT float iree_tokenizer_vocab_token_score(
    const iree_tokenizer_vocab_t* vocab, int32_t token_id) {
  if (!vocab || !vocab->scores || token_id < 0 ||
      token_id > vocab->max_token_id) {
    return 0.0f;
  }
  return vocab->scores[token_id];
}

IREE_API_EXPORT iree_tokenizer_token_attr_t iree_tokenizer_vocab_token_attrs(
    const iree_tokenizer_vocab_t* vocab, int32_t token_id) {
  if (!vocab || token_id < 0 || token_id > vocab->max_token_id) {
    return IREE_TOKENIZER_TOKEN_ATTR_NONE;
  }
  return vocab->tokens[token_id].attributes;
}

IREE_API_EXPORT iree_host_size_t
iree_tokenizer_vocab_capacity(const iree_tokenizer_vocab_t* vocab) {
  // Returns array size (max_token_id + 1) for bounds checking.
  // This may be larger than actual token count for sparse vocabs.
  return vocab ? (iree_host_size_t)(vocab->max_token_id + 1) : 0;
}

IREE_API_EXPORT iree_host_size_t
iree_tokenizer_vocab_token_count(const iree_tokenizer_vocab_t* vocab) {
  return vocab ? vocab->token_count : 0;
}

IREE_API_EXPORT iree_tokenizer_special_ids_t
iree_tokenizer_vocab_special_ids(const iree_tokenizer_vocab_t* vocab) {
  if (!vocab) return iree_tokenizer_special_ids_none();
  return vocab->special_ids;
}

IREE_API_EXPORT iree_host_size_t
iree_tokenizer_vocab_merge_count(const iree_tokenizer_vocab_t* vocab) {
  return vocab ? vocab->merge_count : 0;
}

IREE_API_EXPORT iree_tokenizer_merge_t iree_tokenizer_vocab_merge(
    const iree_tokenizer_vocab_t* vocab, iree_host_size_t rank) {
  iree_tokenizer_merge_t empty = {0, 0};
  if (!vocab || !vocab->merges || rank >= vocab->merge_count) {
    return empty;
  }
  return vocab->merges[rank];
}

IREE_API_EXPORT void iree_tokenizer_vocab_free(iree_tokenizer_vocab_t* vocab) {
  if (!vocab) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = vocab->allocator;

  // Free hash table.
  if (vocab->hash) {
    iree_tokenizer_vocab_hash_free(vocab->hash);
  }

  // Free owned data.
  if (vocab->tokens) {
    iree_allocator_free(allocator, vocab->tokens);
  }
  if (vocab->scores) {
    iree_allocator_free(allocator, vocab->scores);
  }
  if (vocab->string_data) {
    iree_allocator_free(allocator, vocab->string_data);
  }
  if (vocab->merges) {
    iree_allocator_free(allocator, vocab->merges);
  }

  // Free vocab struct.
  iree_allocator_free(allocator, vocab);
  IREE_TRACE_ZONE_END(z0);
}
