// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON parsing for text transforms from HuggingFace tokenizer.json.
//
// Parses the `pre_tokenizer` field from tokenizer.json files into
// iree_tokenizer_text_transform_t instances.
//
// Supported pre_tokenizer types:
//   - BertPreTokenizer: BERT-style whitespace + punctuation splitting
//   - ByteLevel: GPT-2 style byte-to-Unicode mapping
//   - Metaspace: SentencePiece-style space replacement
//   - Whitespace: Simple whitespace splitting
//   - Sequence: Chain of multiple pre-tokenizers
//
// Usage:
//   iree_tokenizer_text_transform_t transform;
//   iree_status_t status = iree_tokenizer_text_transform_parse_json(
//       json_root, allocator, &transform);
//   // ... use transform ...
//   iree_tokenizer_text_transform_deinitialize(&transform);

#ifndef IREE_TOKENIZER_HUGGINGFACE_TRANSFORM_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_TRANSFORM_JSON_H_

#include "iree/tokenizer/transforms/transform.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses a pre_tokenizer object from tokenizer.json into a transform.
//
// |json_root| is the root JSON object (containing "pre_tokenizer" field).
// |allocator| is used for any heap allocations (Sequence children).
// |out_transform| receives the parsed transform on success.
//
// If the pre_tokenizer field is missing or null, returns a NONE transform.
// If the type is unsupported, returns IREE_STATUS_UNIMPLEMENTED.
// If the JSON structure is invalid, returns IREE_STATUS_INVALID_ARGUMENT.
//
// The caller must call iree_tokenizer_text_transform_deinitialize() when done.
iree_status_t iree_tokenizer_text_transform_parse_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform);

// Parses a single pre_tokenizer object (not the root).
// This is useful for parsing nested pre_tokenizers in Sequence.
iree_status_t iree_tokenizer_text_transform_parse_pretokenizer(
    iree_string_view_t pretokenizer_json, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_TRANSFORM_JSON_H_
