// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_DECODER_SEQUENCE_H_
#define IREE_TOKENIZER_DECODER_SEQUENCE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of child decoders in a sequence.
#define IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH 8

// Allocates a Sequence decoder that chains multiple child decoders.
//
// The sequence processes tokens through each child decoder in order:
//   tokens -> child[0] -> child[1] -> ... -> child[n-1] -> output
//
// All children must have IREE_TOKENIZER_DECODER_FLAG_SHRINKING set, enabling
// zero-allocation in-place transformation. The |child_count| must be at most
// IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH.
//
// The sequence takes ownership of child decoders on successful allocation.
// Child decoders are freed when the sequence is destroyed. On allocation
// failure, the caller retains ownership of children.
//
// Returns IREE_STATUS_INVALID_ARGUMENT if |child_count| exceeds the maximum
// depth or any child lacks the SHRINKING flag.
iree_status_t iree_tokenizer_decoder_sequence_allocate(
    iree_tokenizer_decoder_t* const* children, iree_host_size_t child_count,
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_SEQUENCE_H_
