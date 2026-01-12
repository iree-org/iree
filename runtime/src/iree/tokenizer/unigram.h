// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unigram (SentencePiece) tokenization algorithm.
//
// Unigram is a subword tokenization algorithm used by T5, XLM-RoBERTa, and
// other models trained with SentencePiece. Unlike BPE (greedy merging) or
// WordPiece (longest-prefix match), Unigram finds the globally optimal
// tokenization by maximizing the total log probability of the subword sequence.
//
// Each vocabulary entry has an associated score (log probability). The Viterbi
// algorithm finds the segmentation that maximizes the sum of these scores.
//
// Example: "hello" with vocab {"hello": -5.0, "hel": -3.0, "lo": -2.5, ...}
//   Possible segmentations:
//     ["hello"] -> score = -5.0
//     ["hel", "lo"] -> score = -5.5
//   The algorithm picks the segmentation with the highest total score.
//
// Time complexity: O(n * m) where n = input length, m = max token length.

#ifndef IREE_TOKENIZER_UNIGRAM_H_
#define IREE_TOKENIZER_UNIGRAM_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Unigram Encoder State
//===----------------------------------------------------------------------===//

// Opaque Unigram encoder state.
// Holds the score array indexed by token ID for efficient lookup during
// Viterbi decoding. Reusable across multiple encode calls.
typedef struct iree_tokenizer_unigram_state_t iree_tokenizer_unigram_state_t;

// Allocates a Unigram encoder state from a vocabulary with scores.
//
// This prepares the score lookup table needed for Viterbi decoding.
// The state can be reused for multiple encode calls.
//
// |vocab| is a vocabulary containing Unigram tokens.
// |scores| is an array of log probability scores indexed by token ID.
//   Caller retains ownership.
// |score_count| is the number of elements in the scores array. Must match
//   the vocab capacity (iree_tokenizer_vocab_capacity).
// |unk_score| is the score to use for unknown token fallback.
// |allocator| is used for state allocation.
// |out_state| receives the allocated state on success.
iree_status_t iree_tokenizer_unigram_state_allocate(
    const iree_tokenizer_vocab_t* vocab, const float* scores,
    iree_host_size_t score_count, float unk_score, iree_allocator_t allocator,
    iree_tokenizer_unigram_state_t** out_state);

// Frees a Unigram encoder state.
void iree_tokenizer_unigram_state_free(iree_tokenizer_unigram_state_t* state);

//===----------------------------------------------------------------------===//
// Unigram Tokenization
//===----------------------------------------------------------------------===//

// Tokenizes a single word using the Unigram algorithm (Viterbi decoding).
//
// This function takes a pre-tokenized word (output of whitespace/punctuation
// splitting) and breaks it into subword tokens by finding the segmentation
// that maximizes the total log probability.
//
// |state| is the Unigram encoder state (contains vocab and scores).
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
// Unknown characters that don't exist as tokens are handled by emitting the
// unknown token ID if the vocabulary has one configured.
iree_status_t iree_tokenizer_unigram_encode_word(
    const iree_tokenizer_unigram_state_t* state, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count);

//===----------------------------------------------------------------------===//
// Unigram Tokenizer (unified interface)
//===----------------------------------------------------------------------===//

// Allocates a Unigram tokenizer.
//
// This creates a tokenizer using the Unigram algorithm. The returned tokenizer
// implements the common iree_tokenizer_t interface.
//
// |vocab| is consumed by this call (move semantics). On success, the tokenizer
//   owns it and frees it when the tokenizer is freed. On failure, this function
//   frees the vocab before returning. Caller must not use vocab after this
//   call.
// |scores| is consumed by this call (move semantics). The array of scores
//   indexed by token ID. On success, the tokenizer owns it and frees it when
//   the tokenizer is freed. On failure, this function frees it before
//   returning.
// |score_count| is the number of elements in the scores array. Must match
//   the vocab capacity (iree_tokenizer_vocab_capacity).
// |unk_score| is the score to use for unknown token fallback.
// |allocator| is used for state allocation.
// |out_tokenizer| receives the allocated tokenizer on success.
iree_status_t iree_tokenizer_unigram_allocate(iree_tokenizer_vocab_t* vocab,
                                              float* scores,
                                              iree_host_size_t score_count,
                                              float unk_score,
                                              iree_allocator_t allocator,
                                              iree_tokenizer_t** out_tokenizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_UNIGRAM_H_
