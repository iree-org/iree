// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared added_tokens handling for tokenizer.json parsing.
//
// The added_tokens array in tokenizer.json contains special tokens that may
// not be in the model.vocab. This module provides:
// - Temporary storage for added_tokens during parsing
// - Special token pattern matching
// - JSON parsing of the added_tokens array
// - Finalization (add missing tokens to vocab, set special IDs)

#ifndef IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKENS_H_
#define IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKENS_H_

#include "iree/base/api.h"
#include "iree/tokenizer/literals.h"
#include "iree/tokenizer/types.h"
#include "iree/tokenizer/vocab_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Added Token Entry
//===----------------------------------------------------------------------===//

// Entry for a token from the added_tokens array.
typedef struct iree_tokenizer_added_token_entry_t {
  int32_t id;
  char* content;  // Owned, unescaped copy.
  iree_host_size_t content_length;
  bool is_special;
  bool found_in_vocab;  // Set to true when we find this ID in model.vocab.
  iree_tokenizer_special_token_t special_type;  // -1 if not a known special.
} iree_tokenizer_added_token_entry_t;

//===----------------------------------------------------------------------===//
// Added Tokens Storage
//===----------------------------------------------------------------------===//

// Temporary storage for added_tokens during parsing.
typedef struct iree_tokenizer_added_tokens_t {
  iree_tokenizer_added_token_entry_t* entries;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_allocator_t allocator;
} iree_tokenizer_added_tokens_t;

// Initializes added_tokens storage.
void iree_tokenizer_added_tokens_initialize(
    iree_tokenizer_added_tokens_t* added_tokens, iree_allocator_t allocator);

// Frees added_tokens storage and all owned strings.
void iree_tokenizer_added_tokens_deinitialize(
    iree_tokenizer_added_tokens_t* added_tokens);

// Adds an entry to the added_tokens storage.
// Skips duplicates by ID (first entry wins).
iree_status_t iree_tokenizer_added_tokens_add(
    iree_tokenizer_added_tokens_t* added_tokens, int32_t id,
    iree_string_view_t content, bool is_special,
    iree_tokenizer_special_token_t special_type);

// Looks up an added_token entry by ID. Returns NULL if not found.
iree_tokenizer_added_token_entry_t* iree_tokenizer_added_tokens_find(
    iree_tokenizer_added_tokens_t* added_tokens, int32_t id);

// Scans model.vocab for special tokens using pattern matching.
// Used as fallback when added_tokens array is empty/missing.
// Looks up model.unk_token and matches known patterns like [CLS], <s>, etc.
// Sets ATTR_SPECIAL and special token IDs on the vocab builder.
iree_status_t iree_tokenizer_detect_specials_from_vocab(
    iree_string_view_t model, iree_tokenizer_vocab_builder_t* builder);

//===----------------------------------------------------------------------===//
// JSON Parsing
//===----------------------------------------------------------------------===//

// Parses the added_tokens array from tokenizer.json into temporary storage.
// Returns ok if added_tokens is missing (it's optional).
iree_status_t iree_tokenizer_added_tokens_parse_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_added_tokens_t* added_tokens);

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

// Adds any tokens from added_tokens that weren't found in model.vocab,
// and sets special token ID mappings on the builder.
iree_status_t iree_tokenizer_added_tokens_finalize(
    iree_tokenizer_added_tokens_t* added_tokens,
    iree_tokenizer_vocab_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKENS_H_
