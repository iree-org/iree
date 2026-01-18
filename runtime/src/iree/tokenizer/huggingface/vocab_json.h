// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared vocabulary JSON parsing utilities for BPE and WordPiece formats.
//
// Both BPE and WordPiece tokenizer.json files share the same structure for
// vocabulary storage: a "model" object containing a "vocab" dictionary mapping
// token strings to integer IDs. This module provides common parsing functions
// to avoid code duplication.

#ifndef IREE_TOKENIZER_HUGGINGFACE_VOCAB_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_VOCAB_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/huggingface/added_tokens.h"
#include "iree/tokenizer/vocab_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Extracts the "model" object from the tokenizer.json root.
// Returns IREE_STATUS_INVALID_ARGUMENT if the model field is missing or empty.
iree_status_t iree_tokenizer_json_extract_model(iree_string_view_t root,
                                                iree_string_view_t* out_model);

// Validates that model.type matches |expected_type| (e.g., "BPE", "WordPiece").
// Accepts missing type field (inference already happened) or exact match.
// Returns IREE_STATUS_INVALID_ARGUMENT if type is present but doesn't match.
iree_status_t iree_tokenizer_json_validate_model_type(
    iree_string_view_t model, iree_string_view_t expected_type);

// Counts the number of entries in the model.vocab object.
// Returns IREE_STATUS_INVALID_ARGUMENT if vocab field is missing or empty.
iree_status_t iree_tokenizer_json_count_vocab(iree_string_view_t model,
                                              iree_host_size_t* out_count);

// Estimates vocab capacity from JSON size without parsing.
// This avoids a separate counting pass over the vocab - the vocab_builder will
// grow dynamically if the estimate is too low.
//
// Estimation is based on typical vocab entry size:
//   "token_text": 12345,
// Average entry is ~25 bytes, but varies widely:
//   - Short tokens (punctuation): ~10 bytes ("!": 0,)
//   - Long tokens (subwords): ~40+ bytes ("▁something": 12345,)
// Returns at least 1000 even for small JSON inputs.
iree_host_size_t iree_tokenizer_json_estimate_vocab_capacity(
    iree_host_size_t json_size);

// Parses vocab entries from model.vocab into the builder.
// For each entry, validates against |added_tokens| (if non-NULL) to set
// ATTR_SPECIAL on tokens that appear in the added_tokens list.
// Also validates that added_tokens content matches vocab content for shared
// IDs.
iree_status_t iree_tokenizer_json_parse_vocab(
    iree_string_view_t model, iree_allocator_t allocator,
    iree_tokenizer_added_tokens_t* added_tokens,
    iree_tokenizer_vocab_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_VOCAB_JSON_H_
