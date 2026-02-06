// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Builder for constructing vocabularies from external data sources.
//
// Usage:
//   iree_tokenizer_vocab_builder_t* builder = NULL;
//   iree_tokenizer_vocab_builder_allocate(capacity, allocator, &builder);
//   for (...) {
//     iree_tokenizer_vocab_builder_add_token(builder, text, score, attrs);
//   }
//   iree_tokenizer_vocab_builder_set_special_token(builder, type, id);
//   iree_tokenizer_vocab_t* vocab = NULL;
//   iree_tokenizer_vocab_builder_build(builder, &vocab);
//   // builder is consumed; use vocab

#ifndef IREE_TOKENIZER_VOCAB_BUILDER_H_
#define IREE_TOKENIZER_VOCAB_BUILDER_H_

#include "iree/base/api.h"
#include "iree/tokenizer/vocab/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Vocab Builder
//===----------------------------------------------------------------------===//

// Opaque builder type for constructing vocabularies.
typedef struct iree_tokenizer_vocab_builder_t iree_tokenizer_vocab_builder_t;

// Allocates an empty vocab builder.
//
// |capacity_hint| pre-sizes internal arrays to avoid reallocation during
// token addition. Use 0 if the final size is unknown.
// |allocator| is used for all allocations during building.
// |out_builder| receives the allocated builder on success.
iree_status_t iree_tokenizer_vocab_builder_allocate(
    iree_host_size_t capacity_hint, iree_allocator_t allocator,
    iree_tokenizer_vocab_builder_t** out_builder);

// Adds a token to the vocabulary with implicit sequential ID.
//
// Tokens are assigned IDs in the order added (0, 1, 2, ...).
// Use this for sources where tokens are already ordered by ID.
//
// |text| is the token string (copied into the builder's string table).
// |score| is the BPE priority score (use 0.0 for WordPiece/unscored).
// |attrs| are attribute flags for the token.
iree_status_t iree_tokenizer_vocab_builder_add_token(
    iree_tokenizer_vocab_builder_t* builder, iree_string_view_t text,
    float score, iree_tokenizer_token_attr_t attrs);

// Adds a token with an explicit ID (for out-of-order insertion).
//
// Use this when importing from sources where tokens aren't ordered by ID
// (e.g., JSON objects). Can be mixed freely with add_token() - the builder
// automatically sorts tokens by ID during build() when explicit IDs are used.
//
// |token_id| is the target ID for this token.
// |text| is the token string (copied into the builder's string table).
// |score| is the BPE priority score (use 0.0 for WordPiece/unscored).
// |attrs| are attribute flags for the token.
iree_status_t iree_tokenizer_vocab_builder_add_token_with_id(
    iree_tokenizer_vocab_builder_t* builder, int32_t token_id,
    iree_string_view_t text, float score, iree_tokenizer_token_attr_t attrs);

// Sets a special token ID.
//
// Can be called anytime before build(). Multiple calls for the same type
// will overwrite the previous value.
//
// |type| identifies which special token (UNK, BOS, EOS, etc.).
// |token_id| is the ID of the token to mark as special (-1 to clear).
iree_status_t iree_tokenizer_vocab_builder_set_special_token(
    iree_tokenizer_vocab_builder_t* builder,
    iree_tokenizer_special_token_t type, int32_t token_id);

// Sets attributes on an existing token by ID.
//
// Use this to update attributes on a token that was already added.
// This is useful for fallback special token detection where the token is
// added first and then later discovered to be a special token.
//
// |token_id| is the target ID of the token to update.
// |attrs| are attribute flags to OR into the token's existing attributes.
//
// Returns NOT_FOUND if the token ID hasn't been added to the builder yet.
iree_status_t iree_tokenizer_vocab_builder_add_token_attrs(
    iree_tokenizer_vocab_builder_t* builder, int32_t token_id,
    iree_tokenizer_token_attr_t attrs);

// Ensures a token exists with the given ID and attributes.
//
// This is an "upsert" operation:
// - If the token ID already exists: ORs in |attrs| to existing attributes.
// - If the token ID doesn't exist: inserts with given |text|, |score|, |attrs|.
//
// Use this when merging tokens from multiple sources where some tokens may
// already exist in the vocabulary (e.g., special tokens that may or may not
// overlap with the base vocabulary).
//
// |token_id| is the target ID for this token.
// |text| is the token string (only used if inserting; copied into string
// table). |score| is the BPE priority score (only used if inserting). |attrs|
// are attribute flags to set/add on the token.
iree_status_t iree_tokenizer_vocab_builder_ensure_token(
    iree_tokenizer_vocab_builder_t* builder, int32_t token_id,
    iree_string_view_t text, float score, iree_tokenizer_token_attr_t attrs);

// Adds a BPE merge rule.
//
// Merges are stored in the order added (rank order). Index 0 = highest
// priority merge (applied first during encoding).
//
// |left_id| and |right_id| are the token IDs to merge. The result token ID
// is implicitly determined by the vocabulary (the token whose text is the
// concatenation of left and right).
iree_status_t iree_tokenizer_vocab_builder_add_merge(
    iree_tokenizer_vocab_builder_t* builder, uint32_t left_id,
    uint32_t right_id);

// Finalizes the vocabulary and transfers ownership.
//
// This consumes the builder: the builder is freed and must not be used after
// this call. On failure, the builder is also freed.
//
// |out_vocab| receives the constructed vocabulary on success.
iree_status_t iree_tokenizer_vocab_builder_build(
    iree_tokenizer_vocab_builder_t* builder,
    iree_tokenizer_vocab_t** out_vocab);

// Frees a builder without building a vocabulary.
//
// Only call this if build() will not be called (e.g., on error during token
// addition). After build() is called, the builder is already freed.
void iree_tokenizer_vocab_builder_free(iree_tokenizer_vocab_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_VOCAB_BUILDER_H_
