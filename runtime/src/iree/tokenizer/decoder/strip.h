// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_DECODER_STRIP_H_
#define IREE_TOKENIZER_DECODER_STRIP_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum content length supported by Strip decoder.
// Empirical analysis of HuggingFace tokenizers shows all Strip decoders use
// content=" " (1 byte). 8 bytes provides headroom for other patterns.
#define IREE_TOKENIZER_DECODER_STRIP_MAX_CONTENT_LENGTH 8

// Allocates a Strip decoder that removes leading/trailing characters.
//
// Strips up to |start_count| occurrences of |content| from the beginning of
// the output, and up to |stop_count| from the end.
//
// Note: |stop_count| > 0 requires buffering which is not yet supported. Pass 0
// for streaming-compatible behavior.
//
// Empirical analysis of HuggingFace tokenizers (2024-2026) shows all Strip
// decoders use: content=" ", start=1, stop=0 (strip up to 1 leading space).
iree_status_t iree_tokenizer_decoder_strip_allocate(
    iree_string_view_t content, iree_host_size_t start_count,
    iree_host_size_t stop_count, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_STRIP_H_
