// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sequence segmenter for chaining multiple pre-tokenizers.
//
// Chains multiple child segmenters in order, where each child further
// subdivides segments from the previous stage. This matches HuggingFace's
// Sequence pre-tokenizer behavior exactly.
//
// Processing model:
//   text -> child[0] -> child[1] -> ... -> child[n-1] -> segments
//
// Each segment from child[i] becomes independent input for child[i+1].
// Children[1..n-1] see each segment as a complete mini-stream (process +
// finalize per segment), while child[0] maintains true streaming state across
// input chunks.
//
// The implementation uses a hybrid batch/single-segment approach:
//   Batch mode (hot path): Buffers up to 64 segments for efficient batching.
//   Single-segment mode (fallback): Processes one segment at a time when batch
//     buffer would overflow, guaranteeing progress on any valid input.
//
// This design ensures no failure on valid input regardless of segment count
// or expansion ratio through the chain.

#ifndef IREE_TOKENIZER_SEGMENTER_SEQUENCE_H_
#define IREE_TOKENIZER_SEGMENTER_SEQUENCE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of child segmenters in a sequence.
#define IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH 8

// Allocates a Sequence segmenter that chains multiple child segmenters.
//
// The sequence processes text through each child segmenter in order, where each
// child further subdivides segments from the previous stage. This matches
// HuggingFace's Sequence pre-tokenizer behavior exactly.
//
// The |child_count| must be at least 2 (the tokenizer uses single segmenters
// directly rather than wrapping in a sequence) and at most
// IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH.
//
// The sequence takes ownership of child segmenters on successful allocation.
// Child segmenters are freed when the sequence is destroyed. On allocation
// failure, the caller retains ownership of children.
//
// Returns IREE_STATUS_INVALID_ARGUMENT if |child_count| is less than 2, exceeds
// the maximum depth, or any child pointer is NULL.
iree_status_t iree_tokenizer_segmenter_sequence_allocate(
    iree_tokenizer_segmenter_t* const* children, iree_host_size_t child_count,
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_SEQUENCE_H_
