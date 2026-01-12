// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WordPiece tokenization algorithm (BERT-style).
//
// WordPiece is a greedy longest-prefix-first tokenization algorithm used by
// BERT and related models. Given a word, it attempts to find the longest
// matching prefix in the vocabulary, then continues with the remainder
// using a continuation prefix (typically "##").
//
// Example: "unhappiness" -> ["un", "##happi", "##ness"]
//
// The algorithm:
// 1. Try to match the whole word
// 2. If not found, try progressively shorter prefixes
// 3. When a prefix is found, emit it and continue with the suffix
// 4. For suffixes, prepend the continuation prefix ("##")
// 5. If a character cannot be matched, emit [UNK] for the whole word

#ifndef IREE_TOKENIZER_WORDPIECE_H_
#define IREE_TOKENIZER_WORDPIECE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// WordPiece Configuration
//===----------------------------------------------------------------------===//

// Configuration for WordPiece tokenization.
typedef struct iree_tokenizer_wordpiece_config_t {
  // Maximum number of characters per input word. Words longer than this
  // are treated as [UNK]. Set to 0 for default (200).
  //
  // Due to a fixed-size scratch buffer, the maximum supported value is
  // (1024 - prefix_size) / 4 characters (accounting for 4-byte UTF-8).
  // With the default "##" prefix, this is 255 characters. Values exceeding
  // this limit will cause iree_tokenizer_wordpiece_allocate to fail.
  iree_host_size_t max_input_chars_per_word;
  // Continuation prefix for subword tokens. Set data=NULL for default ("##").
  // Set data to non-NULL with size=0 for explicit empty prefix.
  iree_string_view_t continuing_subword_prefix;
} iree_tokenizer_wordpiece_config_t;

// Default WordPiece configuration.
#define IREE_TOKENIZER_WORDPIECE_CONFIG_DEFAULT   \
  ((iree_tokenizer_wordpiece_config_t){           \
      .max_input_chars_per_word = 200,            \
      .continuing_subword_prefix = IREE_SV("##"), \
  })

//===----------------------------------------------------------------------===//
// WordPiece Tokenization
//===----------------------------------------------------------------------===//

// Tokenizes a single word using the WordPiece algorithm.
//
// This function takes a pre-tokenized word (output of whitespace/punctuation
// splitting) and breaks it into subword tokens according to the WordPiece
// algorithm.
//
// |vocab| is a vocabulary containing the WordPiece tokens.
// |config| configures the tokenization (or NULL for defaults).
// |word| is the input word to tokenize (must not contain whitespace).
// |out_ids| receives the output token IDs (caller-allocated).
// |max_ids| is the capacity of |out_ids|.
// |out_count| receives the number of tokens produced.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_RESOURCE_EXHAUSTED if |max_ids| is insufficient
//   - IREE_STATUS_INVALID_ARGUMENT for invalid inputs
//
// If the word cannot be tokenized (no matching subwords), a single [UNK]
// token ID is output if the vocabulary has an unknown token configured.
// Otherwise returns IREE_STATUS_NOT_FOUND.
iree_status_t iree_tokenizer_wordpiece_encode_word(
    const iree_tokenizer_vocab_t* vocab,
    const iree_tokenizer_wordpiece_config_t* config, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count);

//===----------------------------------------------------------------------===//
// WordPiece Tokenizer (unified interface)
//===----------------------------------------------------------------------===//

// Allocates a WordPiece tokenizer.
//
// This creates a tokenizer using the WordPiece algorithm. The returned
// tokenizer implements the common iree_tokenizer_t interface.
//
// |vocab| is consumed by this call (move semantics). On success, the tokenizer
//   owns it and frees it when the tokenizer is freed. On failure, this function
//   frees the vocab before returning. Caller must not use vocab after this
//   call.
// |config| configures the tokenization (or NULL for defaults).
// |prefix_storage| is consumed by this call (move semantics). On success, the
//   tokenizer owns it. On failure, this function frees it before returning.
//   Pass NULL if using static/default storage.
// |allocator| is used for state allocation.
// |out_tokenizer| receives the allocated tokenizer on success.
iree_status_t iree_tokenizer_wordpiece_allocate(
    iree_tokenizer_vocab_t* vocab,
    const iree_tokenizer_wordpiece_config_t* config, char* prefix_storage,
    iree_allocator_t allocator, iree_tokenizer_t** out_tokenizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_WORDPIECE_H_
