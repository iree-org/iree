// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BERT and whitespace text transforms with inline normalization.
//
// BERT: Splits on whitespace, isolates punctuation and CJK characters.
// Whitespace: Simple whitespace splitting only.
//
// Normalization is fused into the character scan loop for zero-allocation
// streaming. Words are accumulated into a stack buffer as they are normalized.

#ifndef IREE_TOKENIZER_TRANSFORMS_BERT_H_
#define IREE_TOKENIZER_TRANSFORMS_BERT_H_

#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/transforms/transform.h"

#ifdef __cplusplus
extern "C" {
#endif

// BERT encode with inline normalization.
// Splits on whitespace, isolates punctuation and CJK characters.
// Normalization (lowercase, strip accents, etc.) is applied per-codepoint
// as the text is scanned. Words are accumulated in a stack buffer.
// Decode is passthrough (handled in transform.c dispatch).
iree_status_t iree_tokenizer_bert_encode(
    const iree_tokenizer_normalizer_t* normalizer, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

// Whitespace encode with inline normalization.
// Splits on whitespace only (punctuation is not isolated).
// Normalization is applied per-codepoint as the text is scanned.
// Decode is passthrough (handled in transform.c dispatch).
iree_status_t iree_tokenizer_whitespace_encode(
    const iree_tokenizer_normalizer_t* normalizer, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TRANSFORMS_BERT_H_
