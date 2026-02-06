// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_MODEL_WORDPIECE_H_
#define IREE_TOKENIZER_MODEL_WORDPIECE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/types.h"
#include "iree/tokenizer/vocab/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// WordPiece Model
//===----------------------------------------------------------------------===//

// Configuration flags for WordPiece tokenization.
typedef enum iree_tokenizer_wordpiece_flag_bits_e {
  IREE_TOKENIZER_WORDPIECE_FLAG_NONE = 0,
} iree_tokenizer_wordpiece_flag_bits_t;
typedef uint32_t iree_tokenizer_wordpiece_flags_t;

// Allocates a WordPiece model from a vocabulary.
//
// WordPiece is a greedy longest-match tokenization algorithm used by
// BERT, DistilBERT, and other models. Given a segment (word):
//   1. Check if the word exceeds max_input_chars_per_word; if so, emit [UNK]
//   2. Find the longest prefix substring in the vocabulary
//   3. For subsequent subwords, prepend the continuing_subword_prefix ("##")
//   4. If any position has no valid substring, emit [UNK] for the entire word
//
// Example: "unaffable" with prefix="##"
//   "unaffable" not in vocab
//   "unaffabl" not in vocab
//   ...
//   "un" in vocab -> emit "un"
//   "##affable" not in vocab
//   "##affabl" not in vocab
//   ...
//   "##aff" in vocab -> emit "##aff"
//   "##able" in vocab -> emit "##able"
//   Result: ["un", "##aff", "##able"]
//
// Pre-computation semantics: Because any sub-token failure causes the entire
// word to become [UNK], WordPiece pre-computes all sub-tokens for a segment
// before emitting any. This ensures correctness when the output buffer fills
// mid-word — we never partially commit a word that would later fail.
//
// |vocab| is the vocabulary containing tokens and the UNK special token ID.
//   The model stores a reference but does not take ownership — the caller
//   must ensure vocab outlives the model.
// |continuing_subword_prefix| is the prefix prepended to non-initial subwords
//   (typically "##"). Maximum length is 16 bytes.
// |max_input_chars_per_word| is the maximum number of Unicode characters
//   per word. Words exceeding this limit are replaced with [UNK].
// |flags| controls tokenization behavior.
// |allocator| is used for model allocation.
// |out_model| receives the allocated model on success.
iree_status_t iree_tokenizer_wordpiece_model_allocate(
    const iree_tokenizer_vocab_t* vocab,
    iree_string_view_t continuing_subword_prefix,
    iree_host_size_t max_input_chars_per_word,
    iree_tokenizer_wordpiece_flags_t flags, iree_allocator_t allocator,
    iree_tokenizer_model_t** out_model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_MODEL_WORDPIECE_H_
