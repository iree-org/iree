// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE (Byte-Pair Encoding) tokenization algorithm.
//
// BPE is an iterative merge-based tokenization algorithm used by GPT-2,
// Llama, and many other models. Given a word:
// 1. Break it into initial symbols (characters or bytes)
// 2. Find all adjacent pairs with merge rules
// 3. Apply the highest priority merge
// 4. Repeat until no more merges apply
//
// Example: "hello" -> ['h','e','l','l','o']
//   merge('l','l') -> ['h','e','ll','o']
//   merge('h','e') -> ['he','ll','o']
//   merge('he','ll') -> ['hell','o']
//   merge('hell','o') -> ['hello']
//
// The vocabulary provides both the token definitions and the merge rules.
// Merge priority is determined by rank (lower rank = higher priority).

#ifndef IREE_TOKENIZER_BPE_H_
#define IREE_TOKENIZER_BPE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// BPE Encoder State
//===----------------------------------------------------------------------===//

// Opaque BPE encoder state.
// Holds the merge lookup table for efficient (left_id, right_id) -> rank
// lookup. Reusable across multiple encode calls for the same vocabulary.
typedef struct iree_tokenizer_bpe_state_t iree_tokenizer_bpe_state_t;

// Allocates a BPE encoder state from a vocabulary.
//
// This builds the merge lookup table needed for efficient encoding.
// The state can be reused for multiple encode calls.
//
// |vocab| is a vocabulary containing BPE tokens and merge rules.
// |allocator| is used for state allocation.
// |out_state| receives the allocated state on success.
iree_status_t iree_tokenizer_bpe_state_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_allocator_t allocator,
    iree_tokenizer_bpe_state_t** out_state);

// Frees a BPE encoder state.
void iree_tokenizer_bpe_state_free(iree_tokenizer_bpe_state_t* state);

// Sets the end-of-word suffix for BPE encoding.
//
// When set, this suffix is appended to each word before tokenization.
// Used by CLIP-style models where tokens like "a</w>" represent word-final
// positions. Maximum suffix length is 16 bytes.
//
// |state| is the BPE encoder state.
// |suffix| is the suffix to append (e.g., "</w>"). Pass empty for none.
iree_status_t iree_tokenizer_bpe_state_set_end_of_word_suffix(
    iree_tokenizer_bpe_state_t* state, iree_string_view_t suffix);

//===----------------------------------------------------------------------===//
// BPE Tokenization
//===----------------------------------------------------------------------===//

// Tokenizes a single word using the BPE algorithm.
//
// This function takes a pre-tokenized word (output of whitespace/punctuation
// splitting) and breaks it into subword tokens according to the BPE algorithm.
//
// |state| is the BPE encoder state (contains vocab and merge lookup).
// |word| is the input word to tokenize (may contain UTF-8).
// |out_ids| receives the output token IDs (caller-allocated).
// |max_ids| is the capacity of |out_ids|.
// |out_count| receives the number of tokens produced.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_RESOURCE_EXHAUSTED if |max_ids| is insufficient
//   - IREE_STATUS_INVALID_ARGUMENT for invalid inputs
//
// Unknown characters that don't exist as tokens are handled by byte fallback
// if the vocabulary contains byte tokens (<0x00> through <0xFF>).
iree_status_t iree_tokenizer_bpe_encode_word(
    const iree_tokenizer_bpe_state_t* state, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count);

//===----------------------------------------------------------------------===//
// BPE Tokenizer (unified interface)
//===----------------------------------------------------------------------===//

// Sets the end-of-word suffix on a BPE tokenizer.
//
// When set, this suffix is appended to each word before tokenization.
// Used by CLIP-style models where tokens like "a</w>" represent word-final
// positions. Maximum suffix length is 16 bytes.
//
// |tokenizer| must be a BPE tokenizer created by iree_tokenizer_bpe_allocate.
// |suffix| is the suffix to append (e.g., "</w>"). Pass empty for none.
iree_status_t iree_tokenizer_bpe_set_end_of_word_suffix(
    iree_tokenizer_t* tokenizer, iree_string_view_t suffix);

// Allocates a BPE tokenizer.
//
// This creates a tokenizer using the BPE algorithm. The returned tokenizer
// implements the common iree_tokenizer_t interface.
//
// |vocab| is consumed by this call (move semantics). On success, the tokenizer
//   owns it and frees it when the tokenizer is freed. On failure, this function
//   frees the vocab before returning. Caller must not use vocab after this
//   call.
// |allocator| is used for state allocation.
// |out_tokenizer| receives the allocated tokenizer on success.
iree_status_t iree_tokenizer_bpe_allocate(iree_tokenizer_vocab_t* vocab,
                                          iree_allocator_t allocator,
                                          iree_tokenizer_t** out_tokenizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_BPE_H_
