// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sequence text transform implementation for chaining multiple transforms.
//
// Sequence transforms apply a series of child transforms in order during encode
// and in reverse order during decode. This matches HuggingFace's Sequence
// pre-tokenizer behavior.
//
// Use iree_tokenizer_text_transform_initialize_sequence() from transform.h to
// create sequence transforms.

#ifndef IREE_TOKENIZER_TRANSFORMS_SEQUENCE_H_
#define IREE_TOKENIZER_TRANSFORMS_SEQUENCE_H_

#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/transforms/transform.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sequence encode with normalizer propagation.
// Applies each child transform in order, passing the sequence's normalizer
// to all children (fused normalization). For each segment from transform N,
// transform N+1 processes it. Single-transform sequences have zero overhead.
iree_status_t iree_tokenizer_sequence_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_sequence_config_t* config, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

// Sequence decode: applies each child transform in reverse order.
// Starting from the last transform, each transform's decode is applied to
// produce intermediate text that feeds into the previous transform's decode.
// Writes directly to out_buffer.
iree_status_t iree_tokenizer_sequence_decode(
    const iree_tokenizer_sequence_config_t* config, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TRANSFORMS_SEQUENCE_H_
