// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Imports Unigram vocabularies from HuggingFace tokenizer.json format.
//
// Supports T5, XLM-RoBERTa, BGE-M3, and similar SentencePiece-based tokenizers.
//
// Usage:
//   iree_string_view_t json = /* load tokenizer.json */;
//   iree_tokenizer_t* tokenizer = NULL;
//   iree_tokenizer_from_unigram_json(json, allocator, &tokenizer);

#ifndef IREE_TOKENIZER_HUGGINGFACE_UNIGRAM_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_UNIGRAM_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Unigram Vocabulary Import
//===----------------------------------------------------------------------===//

// Imports a Unigram vocabulary from HuggingFace tokenizer.json format.
//
// Parses the JSON, extracts the vocab mapping with scores, and builds an
// immutable vocabulary. The JSON must have model.type = "Unigram".
//
// Unlike BPE/WordPiece, Unigram vocab entries have scores (log probabilities).
// The vocab format is: [["token", score], ["token2", score2], ...]
//
// |json| is the JSON contents (tokenizer.json).
// |allocator| is used for the resulting vocab and any temporary allocations.
// |out_vocab| receives the constructed vocabulary on success.
// |out_scores| receives the score array (must be freed by caller on success).
// |out_score_count| receives the number of elements in the scores array.
// |out_unk_score| receives the score for unknown tokens.
iree_status_t iree_tokenizer_vocab_import_unigram_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_vocab_t** out_vocab, float** out_scores,
    iree_host_size_t* out_score_count, float* out_unk_score);

//===----------------------------------------------------------------------===//
// Unigram Tokenizer Factory
//===----------------------------------------------------------------------===//

// Creates a Unigram tokenizer from HuggingFace tokenizer.json format.
//
// This is the main entry point for loading Unigram tokenizers (T5, XLM-RoBERTa,
// BGE-M3, etc.). Parses the complete tokenizer.json including:
//   - Vocabulary with scores (model.vocab)
//   - Pre-tokenizer and normalizer (pre_tokenizer, normalizer)
//   - Decoder (decoder)
//   - Post-processor (post_processor)
//   - Added tokens / literals (added_tokens)
//
// |json| is the complete tokenizer.json contents.
// |allocator| is used for the tokenizer and all internal allocations.
// |out_tokenizer| receives the constructed tokenizer on success.
iree_status_t iree_tokenizer_from_unigram_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_UNIGRAM_JSON_H_
