// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_SEGMENTER_BERT_H_
#define IREE_TOKENIZER_SEGMENTER_BERT_H_

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a BERT pre-tokenizer segmenter that splits on whitespace and
// isolates punctuation characters as individual segments.
//
// Implements HuggingFace's BertPreTokenizer algorithm: split on Unicode
// whitespace (removed) then isolate punctuation characters. Each punctuation
// character becomes its own segment; whitespace is consumed between segments.
//
// Punctuation is defined as ASCII punctuation (ranges 33-47, 58-64, 91-96,
// 123-126) OR Unicode general category P. This matches HuggingFace's
// tokenizers library definition exactly.
//
// Note: CJK character isolation is NOT part of this segmenter. CJK handling is
// performed by the BertNormalizer (handle_chinese_chars=true) which wraps CJK
// characters with spaces before this pre-tokenizer runs.
//
// Example: "Hello, world!" -> segments ["Hello", ",", "world", "!"]
// Example: "don't" -> ["don", "'", "t"]
//
// Streaming Behavior:
// - Decodes UTF-8 codepoints for Unicode whitespace/punctuation detection
// - Emits word segments when terminated by whitespace or punctuation
// - Emits punctuation characters as isolated single-codepoint segments
// - Incomplete UTF-8 sequences at chunk boundaries are deferred to next call
iree_status_t iree_tokenizer_segmenter_bert_allocate(
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_BERT_H_
