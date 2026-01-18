// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Auto-detecting tokenizer factory from HuggingFace tokenizer.json format.
//
// This header provides a single entry point that auto-detects the tokenizer
// type from the JSON and creates the appropriate tokenizer. For explicit type
// selection, use iree_tokenizer_from_bpe_json() or
// iree_tokenizer_from_wordpiece_json() directly.

#ifndef IREE_TOKENIZER_HUGGINGFACE_TOKENIZER_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_TOKENIZER_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a tokenizer from HuggingFace tokenizer.json format with auto-detect.
//
// This function auto-detects the tokenizer type by examining model.type in the
// JSON. If model.type is absent, it infers the type from the model structure:
// - Presence of "merges" array -> BPE
// - Presence of "continuing_subword_prefix" -> WordPiece
//
// Supported model types:
//   - "BPE" (GPT-2, LLaMA, Qwen, etc.)
//   - "WordPiece" (BERT, DistilBERT, etc.)
//
// The returned tokenizer owns the vocab - both are freed on
// iree_tokenizer_free().
//
// |json| is the JSON contents (tokenizer.json).
// |allocator| is used for the tokenizer and vocab.
// |out_tokenizer| receives the allocated tokenizer on success.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_UNIMPLEMENTED for unsupported tokenizer types
//   - IREE_STATUS_INVALID_ARGUMENT if type cannot be determined
IREE_API_EXPORT iree_status_t iree_tokenizer_from_huggingface_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_TOKENIZER_JSON_H_
