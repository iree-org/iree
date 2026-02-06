// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HuggingFace tokenizer.json normalizer section parser.
//
// This header provides parsing for the "normalizer" object in HuggingFace
// tokenizer.json files. The normalizer section configures text normalization
// applied before tokenization (lowercasing, accent stripping, etc.).
//
// Supported normalizer types:
//   - Sequence: Chains multiple normalizers in order
//   - Lowercase: Unicode case folding
//   - Strip: Leading/trailing whitespace removal
//   - Prepend: Prefix string insertion
//   - StripAccents: Removes combining marks (without NFD)
//   - BertNormalizer: Combined BERT normalization pipeline

#ifndef IREE_TOKENIZER_FORMAT_HUGGINGFACE_NORMALIZER_JSON_H_
#define IREE_TOKENIZER_FORMAT_HUGGINGFACE_NORMALIZER_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Normalizer Parser
//===----------------------------------------------------------------------===//

// Parses a normalizer from its JSON value (the object containing "type").
//
// Dispatches to type-specific parsers based on the "type" field. For Sequence
// normalizers, recursively parses children and compacts out NULL (no-op)
// children. A sequence with 0 children after compaction returns NULL, and a
// sequence with 1 child returns that child directly (no wrapper).
//
// The |normalizer_value| is the JSON text of the normalizer object itself
// (e.g., '{"type":"Lowercase"}'), not the full tokenizer.json file.
// Pass "null" to get a NULL (passthrough) normalizer.
//
// Returns:
//   - IREE_STATUS_OK on success (out_normalizer may be NULL for passthrough)
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON
//   - IREE_STATUS_UNIMPLEMENTED for unsupported normalizer types
iree_status_t iree_tokenizer_huggingface_parse_normalizer(
    iree_string_view_t normalizer_value, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_HUGGINGFACE_NORMALIZER_JSON_H_
