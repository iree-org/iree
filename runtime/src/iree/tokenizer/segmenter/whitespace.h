// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whitespace segmenter for pre-tokenization.
//
// Splits normalized text on whitespace boundaries. Each non-whitespace run
// becomes a segment. Whitespace between segments is consumed but not emitted
// as segments.
//
// Example: "hello world" -> segments ["hello", "world"]
//
// This is the simplest pre-tokenizer, suitable for BPE models that handle
// word boundaries during merging. For more sophisticated splitting
// (punctuation, special characters), use other segmenter implementations.
//
// Streaming Behavior:
// - Scans input byte-by-byte for whitespace
// - Emits segments as soon as whitespace boundary is found
// - Buffers segment start position for partial processing
// - finalize() emits final segment if input didn't end with whitespace

#ifndef IREE_TOKENIZER_SEGMENTER_WHITESPACE_H_
#define IREE_TOKENIZER_SEGMENTER_WHITESPACE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a whitespace segmenter that splits on whitespace boundaries.
// Non-whitespace runs become segments; whitespace is consumed between them.
iree_status_t iree_tokenizer_segmenter_whitespace_allocate(
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_WHITESPACE_H_
