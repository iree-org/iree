// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HuggingFace tokenizer.json parser for tokenizer.
//
// This header provides the main entry point for parsing HuggingFace tokenizer
// JSON format files. The parser is designed for 100% compatibility with the
// HuggingFace `tokenizers` Rust library.
//
// Key features:
// - Strict field validation: unknown fields cause immediate failure
// - Auto-detects model type from model.type or infers from structure
// - Supports BPE, WordPiece, and Unigram models
// - Full support for normalizers, pre-tokenizers, decoders, post-processors
//
// Example usage:
//   iree_tokenizer_t* tokenizer = NULL;
//   iree_status_t status = iree_tokenizer_from_huggingface_json(
//       json_contents, allocator, &tokenizer);
//   if (iree_status_is_ok(status)) {
//     // Use tokenizer for encode/decode
//     iree_tokenizer_free(tokenizer);
//   }

#ifndef IREE_TOKENIZER_FORMAT_HUGGINGFACE_TOKENIZER_JSON_H_
#define IREE_TOKENIZER_FORMAT_HUGGINGFACE_TOKENIZER_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Top-Level Tokenizer Parser
//===----------------------------------------------------------------------===//

// Creates a tokenizer from HuggingFace tokenizer.json format.
//
// This function auto-detects the tokenizer type by examining model.type in the
// JSON. If model.type is absent, it infers the type from the model structure:
// - Presence of "merges" array -> BPE
// - Presence of "continuing_subword_prefix" -> WordPiece
// - Presence of "unk_id" (integer) -> Unigram
//
// Supported model types:
//   - "BPE" (GPT-2, LLaMA, Qwen, etc.)
//   - "WordPiece" (BERT, DistilBERT, etc.)
//   - "Unigram" (SentencePiece-based models like T5, ALBERT)
//
// The returned tokenizer owns all parsed components (normalizer, segmenter,
// model, decoder) and is freed via iree_tokenizer_free().
//
// |json| is the full tokenizer.json contents.
// |allocator| is used for all allocations; stored for cleanup.
// |out_tokenizer| receives the allocated tokenizer on success.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON or missing required
//     fields
//   - IREE_STATUS_UNIMPLEMENTED for unsupported model/component types
//   - IREE_STATUS_OUT_OF_RANGE for allocation failures
iree_status_t iree_tokenizer_from_huggingface_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer);

//===----------------------------------------------------------------------===//
// Builder-Based Parser (Advanced)
//===----------------------------------------------------------------------===//

// Parses tokenizer.json and populates a builder.
//
// This is the lower-level API that allows callers to inspect or modify the
// builder before calling iree_tokenizer_builder_build(). Use this when you
// need to customize the tokenizer after parsing.
//
// The builder must be initialized before calling this function. On success,
// the builder owns all parsed components. On failure, the builder is left in
// a valid but unspecified state (callers should deinitialize it).
//
// |json| is the full tokenizer.json contents.
// |builder| is the pre-initialized builder to populate.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON or missing required
//     fields
//   - IREE_STATUS_UNIMPLEMENTED for unsupported model/component types
iree_status_t iree_tokenizer_parse_huggingface_json(
    iree_string_view_t json, iree_tokenizer_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_HUGGINGFACE_TOKENIZER_JSON_H_
