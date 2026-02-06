// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_MODEL_UNIGRAM_H_
#define IREE_TOKENIZER_MODEL_UNIGRAM_H_

#include "iree/base/api.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/types.h"
#include "iree/tokenizer/vocab/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Unigram Model
//===----------------------------------------------------------------------===//

// Configuration flags for Unigram tokenization.
typedef enum iree_tokenizer_unigram_flag_bits_e {
  IREE_TOKENIZER_UNIGRAM_FLAG_NONE = 0,
  // Disable <0xXX> byte fallback for unreachable segments. When set, segments
  // with no valid tokenization emit a single [UNK] token. When clear (default),
  // unreachable bytes are encoded as individual <0xXX> tokens.
  IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK = 1u << 0,
  // Disable consecutive UNK fusion. When set, each UNK-producing character
  // emits a separate [UNK] token. When clear (default), adjacent UNKs are
  // merged into a single [UNK] (matching SentencePiece behavior).
  IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK = 1u << 1,
} iree_tokenizer_unigram_flag_bits_t;
typedef uint32_t iree_tokenizer_unigram_flags_t;

// Allocates a Unigram (SentencePiece) model from a vocabulary.
//
// Unigram uses Viterbi dynamic programming to find the segmentation with the
// highest total log-probability. Unlike BPE (merge-based) or WordPiece (greedy
// longest-match), Unigram considers ALL valid segmentations and picks the
// optimal one.
//
// Algorithm:
//   For each segment (word):
//     1. Forward pass: at each UTF-8 character boundary, walk a prefix trie to
//        find all matching vocab tokens. Update DP table with:
//          best_score[end] = max(best_score[start] + token_score)
//        If no single-character token covers a position, insert UNK candidate.
//     2. If dp[segment_end] is unreachable: try byte fallback (<0xXX> tokens)
//        or emit a single [UNK].
//     3. Backtrack from end to reconstruct the optimal tokenization.
//     4. During backtrack: expand UNK to byte tokens (if enabled), fuse
//        consecutive UNKs (if enabled).
//
// Streaming: Unigram processes segments incrementally in max_token_length-byte
// chunks using sliding-window Viterbi DP. Each chunk runs exact Viterbi
// (forward pass + backtrack), emitting tokens immediately. This enables
// partial segment processing for split=false models where the entire input
// is one segment — the model emits frozen tokens and reclaims bytes to
// unblock the pipeline's ring buffer. DP buffers are sized to
// max_token_length from the vocabulary, giving O(L) state.
//
// |vocab| is the vocabulary containing tokens with scores. The model stores
//   a reference but does not take ownership — the caller must ensure vocab
//   outlives the model. A prefix trie is built from the vocab at allocation
//   time for efficient DP traversal.
// |unk_token_id| is the resolved UNK token ID (from JSON's unk_id field).
//   Pass IREE_TOKENIZER_TOKEN_ID_INVALID if no UNK token is configured
//   (byte_fallback must then be enabled, or encoding may fail).
// |unk_score| is the log-probability score used when inserting UNK candidates
//   in the Viterbi forward pass. Typically the score of the UNK token itself
//   (e.g., -10.0f). Lower values make UNK less likely to be chosen.
// |flags| controls byte_fallback and UNK fusion behavior.
// |allocator| is used for model and trie allocation.
// |out_model| receives the allocated model on success.
iree_status_t iree_tokenizer_unigram_model_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_token_id_t unk_token_id,
    float unk_score, iree_tokenizer_unigram_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_MODEL_UNIGRAM_H_
