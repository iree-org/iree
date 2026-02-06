// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HuggingFace tokenizer.json model section parser.
//
// This header provides parsing for the "model" object in HuggingFace
// tokenizer.json files. The model section contains the vocabulary and
// algorithm-specific configuration (BPE merges, WordPiece prefixes, etc.).
//
// Supported model types:
//   - BPE: Full implementation with vocab, merges, and all flags
//   - WordPiece: Full implementation with vocab, unk_token, prefix, max chars
//   - Unigram: Full implementation with vocab array, unk_id, byte_fallback

#ifndef IREE_TOKENIZER_FORMAT_HUGGINGFACE_MODEL_JSON_H_
#define IREE_TOKENIZER_FORMAT_HUGGINGFACE_MODEL_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/format/huggingface/added_tokens_json.h"
#include "iree/tokenizer/format/huggingface/types.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/model/unigram.h"
#include "iree/tokenizer/types.h"
#include "iree/tokenizer/vocab/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// BPE Model Parser
//===----------------------------------------------------------------------===//

// Parses a BPE model section from HuggingFace tokenizer.json.
//
// The BPE model contains:
//   - vocab: Object mapping token strings to IDs
//   - merges: Array of merge rules (string "a b" or tuple ["a","b"] format)
//   - Optional flags: byte_fallback, fuse_unk, ignore_merges
//   - Optional affixes: continuing_subword_prefix, end_of_word_suffix
//
// This function performs two-pass parsing:
//   1. Parse vocab into builder, build temporary vocab for merge lookup
//   2. Parse merges using temp vocab for ID resolution, rebuild with merges
//
// Parses a BPE model from |model_json| (the "model" section, not full file),
// using |added_tokens| for special token marking. The |extra_flags| are ORed
// with flags parsed from the JSON (e.g., BYTE_LEVEL_INPUT from pre_tokenizer).
// On success, |out_model| and |out_vocab| receive the parsed model and vocab;
// the model references but does not own the vocab, so vocab must outlive it.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON
//   - IREE_STATUS_NOT_FOUND if merge references unknown token
//   - IREE_STATUS_OUT_OF_RANGE for token IDs out of valid range
iree_status_t iree_tokenizer_huggingface_parse_bpe_model(
    iree_string_view_t model_json,
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_tokenizer_bpe_flags_t extra_flags, iree_allocator_t allocator,
    iree_tokenizer_model_t** out_model, iree_tokenizer_vocab_t** out_vocab);

//===----------------------------------------------------------------------===//
// WordPiece Model Parser
//===----------------------------------------------------------------------===//

// Parses a WordPiece model section from HuggingFace tokenizer.json.
//
// The WordPiece model contains:
//   - vocab: Object mapping token strings to IDs
//   - unk_token: Unknown token text (default "[UNK]")
//   - continuing_subword_prefix: Prefix for non-initial subwords (default "##")
//   - max_input_chars_per_word: Max Unicode chars per word (default 100)
//
// Parses a WordPiece model from |model_json| (the "model" section, not the full
// file), using |added_tokens| for special token marking. On success,
// |out_model| and |out_vocab| receive the parsed model and vocab; the model
// references but does not own the vocab, so vocab must outlive it.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON or invalid parameters
//   - IREE_STATUS_NOT_FOUND if unk_token is not in the vocabulary
//   - IREE_STATUS_OUT_OF_RANGE for token IDs out of valid range
iree_status_t iree_tokenizer_huggingface_parse_wordpiece_model(
    iree_string_view_t model_json,
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model,
    iree_tokenizer_vocab_t** out_vocab);

//===----------------------------------------------------------------------===//
// Unigram Model Parser
//===----------------------------------------------------------------------===//

// Parses a Unigram (SentencePiece) model section from HuggingFace
// tokenizer.json.
//
// The Unigram model contains:
//   - vocab: Array of [token_string, score] tuples
//   - unk_id: Index of UNK token in vocab (default -1 for none)
//   - byte_fallback: Whether to use <0xXX> tokens for unknowns (default true)
//
// Parses a Unigram model from |model_json| (the "model" section, not the full
// file), using |added_tokens| for special token marking. On success,
// |out_model| and |out_vocab| receive the parsed model and vocab; the model
// references but does not own the vocab, so vocab must outlive it.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON or invalid vocab entries
//   - IREE_STATUS_OUT_OF_RANGE for token IDs out of valid range
iree_status_t iree_tokenizer_huggingface_parse_unigram_model(
    iree_string_view_t model_json,
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model,
    iree_tokenizer_vocab_t** out_vocab);

//===----------------------------------------------------------------------===//
// Generic Model Parser
//===----------------------------------------------------------------------===//

// Parses the model section from a HuggingFace tokenizer.json, auto-detecting
// the model type from the JSON structure (explicit "type" field, or inferred
// from presence of "merges" for BPE, "continuing_subword_prefix" for WordPiece,
// or "unk_id" for Unigram).
//
// The |model_json| is the model object's JSON text (the value of the "model"
// key, not the full file). The |pre_tokenizer_flags| carry format-level
// properties discovered during pre_tokenizer parsing (e.g., ByteLevel
// presence) that affect model configuration. The |added_tokens| are used for
// special token attribute marking.
//
// On success, |out_model| and |out_vocab| receive the parsed model and
// vocabulary; the model references but does not own the vocab, so vocab must
// outlive the model.
iree_status_t iree_tokenizer_huggingface_parse_model(
    iree_string_view_t model_json,
    const iree_tokenizer_huggingface_added_tokens_t* added_tokens,
    iree_tokenizer_huggingface_pre_tokenizer_flags_t pre_tokenizer_flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model,
    iree_tokenizer_vocab_t** out_vocab);

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Estimates vocab builder capacity from JSON size.
//
// Uses heuristic: json_size / 40 bytes per entry (average token + overhead).
// Returns at least 1000 for small files.
//
// This helps avoid reallocation during vocab building for large files.
iree_host_size_t iree_tokenizer_huggingface_estimate_vocab_capacity(
    iree_host_size_t json_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_HUGGINGFACE_MODEL_JSON_H_
