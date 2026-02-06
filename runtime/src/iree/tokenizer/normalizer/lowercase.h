// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_LOWERCASE_H_
#define IREE_TOKENIZER_NORMALIZER_LOWERCASE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a lowercase normalizer that converts text to lowercase.
//
// Uses Unicode simple case folding (not locale-specific). ASCII characters
// are processed via a fast path; non-ASCII uses full Unicode lowercasing.
//
// The only Unicode codepoint that expands during lowercasing is U+0130
// (Latin Capital Letter I With Dot Above, İ) which becomes U+0069 + U+0307
// (i + combining dot above). This expansion is handled correctly.
iree_status_t iree_tokenizer_normalizer_lowercase_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_LOWERCASE_H_
