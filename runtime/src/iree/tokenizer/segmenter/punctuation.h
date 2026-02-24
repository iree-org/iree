// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_SEGMENTER_PUNCTUATION_H_
#define IREE_TOKENIZER_SEGMENTER_PUNCTUATION_H_

#include "iree/base/api.h"
#include "iree/tokenizer/regex/exec.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// iree_tokenizer_segmenter_punctuation_t
//===----------------------------------------------------------------------===//

// Allocates a segmenter that splits text on punctuation characters.
// The |behavior| controls how punctuation matches are converted to segments.
//
// Splits text on punctuation characters with configurable split behavior.
// Unlike BertPreTokenizer, this only splits on punctuation — whitespace is
// treated as regular text (included in non-punctuation segments).
//
// Punctuation is defined as ASCII punctuation (ranges 33-47, 58-64, 91-96,
// 123-126) OR Unicode general category P. This matches HuggingFace's
// tokenizers library definition exactly (char::is_ascii_punctuation || Unicode
// P).
//
// The behavior parameter controls how punctuation delimiters interact with
// surrounding text:
//
//   ISOLATED (default): Each punctuation char is its own segment.
//     "Hey friend!" -> ["Hey friend", "!"]
//
//   REMOVED: Punctuation is discarded.
//     "Hey friend!" -> ["Hey friend"]
//
//   MERGED_WITH_PREVIOUS: Punctuation appended to preceding segment.
//     "a.b" -> ["a.", "b"]
//
//   MERGED_WITH_NEXT: Punctuation prepended to following segment.
//     "a.b" -> ["a", ".b"]
//
//   CONTIGUOUS: Consecutive punctuation chars grouped into one segment.
//     "a...b" -> ["a", "...", "b"]
//
// Streaming Behavior:
// - Decodes UTF-8 codepoints for Unicode punctuation detection
// - Emits segments as boundaries are identified
// - Incomplete UTF-8 sequences at chunk boundaries are deferred to next call
iree_status_t iree_tokenizer_segmenter_punctuation_allocate(
    iree_tokenizer_regex_split_behavior_t behavior, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_PUNCTUATION_H_
