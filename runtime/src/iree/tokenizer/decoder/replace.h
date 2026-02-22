// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_DECODER_REPLACE_H_
#define IREE_TOKENIZER_DECODER_REPLACE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a Replace decoder that substitutes all occurrences of |pattern|
// with |content| in decoded token strings.
//
// The pattern must not be empty and must be at least as long as the content
// (shrinking only). This enables zero-allocation in-place transformation.
// Real HuggingFace configs use patterns like "▁" (3 bytes) -> " " (1 byte).
//
// Returns IREE_STATUS_INVALID_ARGUMENT if pattern is empty or expanding.
iree_status_t iree_tokenizer_decoder_replace_allocate(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_REPLACE_H_
