// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Common types shared across tokenizer pipeline stages.
//
// This header defines types used by multiple pipeline components (normalizer,
// segmenter, model, decoder) to avoid circular dependencies.
//
// Adding types to this file is an anti-pattern. This file exists only for
// fundamental types (token IDs, offsets) that must be shared across many
// disparate APIs. Format-specific types (HuggingFace, SentencePiece, etc.)
// belong in their respective format/ subdirectories, not here.

#ifndef IREE_TOKENIZER_TYPES_H_
#define IREE_TOKENIZER_TYPES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Token IDs
//===----------------------------------------------------------------------===//

// Token ID type. Valid token IDs are non-negative integers corresponding to
// vocabulary indices. The value IREE_TOKENIZER_TOKEN_ID_INVALID (-1) indicates
// "not found" or "not present" and must be checked before using as an index.
typedef int32_t iree_tokenizer_token_id_t;

// Sentinel value indicating an invalid or missing token ID.
#define IREE_TOKENIZER_TOKEN_ID_INVALID ((iree_tokenizer_token_id_t) - 1)

// List of token IDs (const pointer + count).
// Used for decode input where we pass a sequence of tokens to decode.
typedef struct iree_tokenizer_token_id_list_t {
  iree_host_size_t count;
  const iree_tokenizer_token_id_t* values;
} iree_tokenizer_token_id_list_t;

// Returns an empty token ID list.
static inline iree_tokenizer_token_id_list_t iree_tokenizer_token_id_list_empty(
    void) {
  iree_tokenizer_token_id_list_t list = {0, NULL};
  return list;
}

// Creates a token ID list from a pointer and count.
static inline iree_tokenizer_token_id_list_t iree_tokenizer_make_token_id_list(
    const iree_tokenizer_token_id_t* values, iree_host_size_t count) {
  iree_tokenizer_token_id_list_t list = {count, values};
  return list;
}

//===----------------------------------------------------------------------===//
// Offset Tracking
//===----------------------------------------------------------------------===//

// Offset pair mapping a token back to original input bytes.
typedef struct iree_tokenizer_offset_t {
  // Start byte in original input.
  iree_host_size_t start;
  // End byte (exclusive) in original input.
  iree_host_size_t end;
} iree_tokenizer_offset_t;

//===----------------------------------------------------------------------===//
// Token Output
//===----------------------------------------------------------------------===//

// Output buffer for token encoding operations.
// Groups capacity, token IDs array, and optional auxiliary arrays together.
// All optional arrays (if non-NULL) must have the same capacity as token_ids,
// enforced by using a single capacity field.
typedef struct iree_tokenizer_token_output_t {
  // Maximum number of tokens that can be written.
  iree_host_size_t capacity;
  // Output array for token IDs (must have capacity elements).
  iree_tokenizer_token_id_t* token_ids;
  // Optional output array for byte offsets into original input (NULL to skip,
  // otherwise must have capacity elements).
  iree_tokenizer_offset_t* token_offsets;
  // Optional output array for type IDs / segment IDs (NULL to skip, otherwise
  // must have capacity elements). Used by post-processors to indicate which
  // sequence each token belongs to (e.g., 0 for sentence A, 1 for sentence B).
  uint8_t* type_ids;
} iree_tokenizer_token_output_t;

// Creates a token output buffer. Pass NULL for optional arrays to skip them.
static inline iree_tokenizer_token_output_t iree_tokenizer_make_token_output(
    iree_tokenizer_token_id_t* token_ids,
    iree_tokenizer_offset_t* token_offsets, uint8_t* type_ids,
    iree_host_size_t capacity) {
  iree_tokenizer_token_output_t output = {capacity, token_ids, token_offsets,
                                          type_ids};
  return output;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TYPES_H_
