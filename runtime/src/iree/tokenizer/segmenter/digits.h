// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Digits segmenter for pre-tokenization.
//
// Splits text on digit boundaries. ASCII digits (0-9) are isolated from
// non-digit text. The `individual_digits` parameter controls whether each
// digit is its own segment or contiguous digits are grouped.
//
// Examples with individual_digits=true:
//   "abc123def" -> ["abc", "1", "2", "3", "def"]
//   "test42" -> ["test", "4", "2"]
//
// Examples with individual_digits=false:
//   "abc123def" -> ["abc", "123", "def"]
//   "test42" -> ["test", "42"]
//
// Unlike whitespace segmentation, all characters become segments - nothing is
// discarded. Empty segments are not produced.

#ifndef IREE_TOKENIZER_SEGMENTER_DIGITS_H_
#define IREE_TOKENIZER_SEGMENTER_DIGITS_H_

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a digits segmenter that splits on digit boundaries.
// If individual_digits is true, each digit becomes its own segment.
// If false, contiguous digit sequences form a single segment.
iree_status_t iree_tokenizer_segmenter_digits_allocate(
    bool individual_digits, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_DIGITS_H_
