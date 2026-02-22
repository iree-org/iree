// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Metaspace segmenter for SentencePiece-style pre-tokenization.
//
// Splits text on a delimiter character (default U+2581, LOWER ONE EIGHTH BLOCK)
// using MergedWithNext behavior: each segment includes the delimiter that
// precedes it. This matches HuggingFace tokenizers' Metaspace pre-tokenizer.
//
// Example: "▁Hey▁friend▁" -> segments ["▁Hey", "▁friend", "▁"]
//
// The segmenter assumes the input has already been normalized (spaces replaced
// with the delimiter character, delimiter prepended if needed). Use this with
// the Metaspace normalizer for complete HuggingFace compatibility.
//
// Streaming Behavior:
// - Scans input for delimiter character (multi-byte UTF-8 aware)
// - Emits segments where each delimiter merges with following text
// - Handles chunk boundaries that split the delimiter's UTF-8 sequence
// - finalize() emits final segment (may be just the delimiter itself)

#ifndef IREE_TOKENIZER_SEGMENTER_METASPACE_H_
#define IREE_TOKENIZER_SEGMENTER_METASPACE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Default replacement character: U+2581 (LOWER ONE EIGHTH BLOCK).
// This is the standard SentencePiece word boundary marker.
#define IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT 0x2581

// Allocates a Metaspace segmenter that splits on a replacement character.
//
// The segmenter implements MergedWithNext splitting: each segment includes
// the delimiter that precedes it (matching HuggingFace tokenizers behavior).
//
// |replacement_codepoint| is the Unicode codepoint to split on.
//   Use 0 for default (U+2581, LOWER ONE EIGHTH BLOCK).
// |split_enabled| controls splitting behavior:
//   true: split on delimiter (normal operation)
//   false: entire input is one segment (delimiter ignored)
// |allocator| is used for the segmenter object allocation.
// |out_segmenter| receives the allocated segmenter on success.
iree_status_t iree_tokenizer_segmenter_metaspace_allocate(
    uint32_t replacement_codepoint, bool split_enabled,
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_METASPACE_H_
