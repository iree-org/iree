// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_SEQUENCE_H_
#define IREE_TOKENIZER_NORMALIZER_SEQUENCE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of child normalizers in a sequence.
// Set to 16 to support complex normalizer pipelines like ALBERT which has 9
// children after flattening nested sequences.
#define IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH 16

// Allocates a Sequence normalizer that chains multiple child normalizers.
// The sequence processes text through each child normalizer in order, piping
// output from each as input to the next. This matches HuggingFace's Sequence
// normalizer behavior exactly.
//
// Operates by chaining multiple child normalizers in order, piping output from
// each as input to the next. This enables composition of simple normalizers
// into complex pipelines (e.g., NFC -> Lowercase -> Strip).
//
// Processing model:
//
//   text -> [child 0] -> [child 1] -> ... -> [child n-1] -> output
//
// The implementation uses intermediate buffering to pipe each child's output
// to the next child's input in the pull-based streaming model.
//
// The |child_count| may be 0 (empty sequence acts as identity/passthrough) and
// must be at most IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH.
//
// The sequence takes ownership of child normalizers on successful allocation.
// Child normalizers are freed when the sequence is destroyed. On allocation
// failure, the caller retains ownership of children.
//
// Returns IREE_STATUS_INVALID_ARGUMENT if |child_count| exceeds the maximum
// depth or any child pointer is NULL.
iree_status_t iree_tokenizer_normalizer_sequence_allocate(
    iree_tokenizer_normalizer_t* const* children, iree_host_size_t child_count,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_SEQUENCE_H_
