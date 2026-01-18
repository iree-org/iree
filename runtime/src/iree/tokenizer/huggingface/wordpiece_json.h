// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Imports WordPiece vocabularies from HuggingFace tokenizer.json format.
//
// Supports BERT, DistilBERT, and similar WordPiece-based tokenizers.
//
// Usage:
//   iree_string_view_t json = /* load tokenizer.json */;
//   iree_tokenizer_vocab_t* vocab = NULL;
//   iree_tokenizer_vocab_import_wordpiece_json(json, allocator, &vocab);

#ifndef IREE_TOKENIZER_HUGGINGFACE_WORDPIECE_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_WORDPIECE_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

// Imports a WordPiece vocabulary from HuggingFace tokenizer.json format.
//
// Parses the JSON, extracts the vocab mapping and special tokens, and builds
// an immutable vocabulary. The JSON must have model.type = "WordPiece".
//
// The vocab object maps token strings to integer IDs (0, 1, 2, ...). Special
// tokens like [UNK], [CLS], [SEP], [PAD], [MASK] are automatically detected
// from the added_tokens array.
//
// |json| is the JSON contents (tokenizer.json).
// |allocator| is used for the resulting vocab and any temporary allocations.
// |out_vocab| receives the constructed vocabulary on success.
iree_status_t iree_tokenizer_vocab_import_wordpiece_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_vocab_t** out_vocab);

//===----------------------------------------------------------------------===//
// WordPiece Tokenizer Factory
//===----------------------------------------------------------------------===//

// Creates a WordPiece tokenizer from HuggingFace tokenizer.json format.
//
// This is the main entry point for loading WordPiece tokenizers (BERT,
// DistilBERT, etc.). Parses the complete tokenizer.json including:
//   - Vocabulary (model.vocab)
//   - Pre-tokenizer and normalizer (pre_tokenizer, normalizer)
//   - Decoder (decoder)
//   - Post-processor (post_processor)
//   - Added tokens / literals (added_tokens)
//   - WordPiece-specific config (continuing_subword_prefix,
//   max_input_chars_per_word)
//
// |json| is the complete tokenizer.json contents.
// |allocator| is used for the tokenizer and all internal allocations.
// |out_tokenizer| receives the constructed tokenizer on success.
iree_status_t iree_tokenizer_from_wordpiece_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_WORDPIECE_JSON_H_
