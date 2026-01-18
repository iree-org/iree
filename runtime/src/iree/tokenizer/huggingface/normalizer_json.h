// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON parsing for text normalizers from HuggingFace tokenizer.json.
//
// Parses the `normalizer` field from tokenizer.json files into
// iree_tokenizer_normalizer_t instances.
//
// Supported normalizer types:
//   - BertNormalizer: BERT-style normalization (lowercase, strip accents, etc.)
//   - Lowercase: Simple lowercasing
//   - StripAccents: Remove diacritical marks (NFD decomposition)
//   - Sequence: Chain of multiple normalizers
//
// Usage:
//   iree_tokenizer_normalizer_t normalizer;
//   iree_status_t status = iree_tokenizer_normalizer_parse_json(
//       json_root, allocator, &normalizer);
//   // ... use normalizer ...
//   iree_tokenizer_normalizer_deinitialize(&normalizer);

#ifndef IREE_TOKENIZER_HUGGINGFACE_NORMALIZER_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_NORMALIZER_JSON_H_

#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses a normalizer object from tokenizer.json into a normalizer.
//
// |json_root| is the root JSON object (containing "normalizer" field).
// |allocator| is used for any heap allocations (Sequence children).
// |out_normalizer| receives the parsed normalizer on success.
//
// If the normalizer field is missing or null, returns a NONE normalizer.
// If the type is unsupported, returns IREE_STATUS_UNIMPLEMENTED.
// If the JSON structure is invalid, returns IREE_STATUS_INVALID_ARGUMENT.
//
// The caller must call iree_tokenizer_normalizer_deinitialize() when done.
iree_status_t iree_tokenizer_normalizer_parse_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer);

// Parses a single normalizer object (not the root).
// This is useful for parsing nested normalizers in Sequence.
iree_status_t iree_tokenizer_normalizer_parse_normalizer_field(
    iree_string_view_t normalizer_json, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t* out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_NORMALIZER_JSON_H_
