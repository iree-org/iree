// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_STRIP_H_
#define IREE_TOKENIZER_NORMALIZER_STRIP_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a strip normalizer that removes leading and/or trailing whitespace.
//
// Uses Unicode White_Space property for whitespace detection, which includes:
// - ASCII whitespace: space, tab, LF, VT, FF, CR (0x09-0x0D, 0x20)
// - Unicode whitespace: NBSP (U+00A0), En/Em spaces (U+2000-U+200A),
//   ideographic space (U+3000), etc.
//
// Parameters:
//   strip_left: If true, removes leading whitespace.
//   strip_right: If true, removes trailing whitespace.
//   allocator: Allocator for the normalizer.
//   out_normalizer: Receives the created normalizer.
iree_status_t iree_tokenizer_normalizer_strip_allocate(
    bool strip_left, bool strip_right, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_STRIP_H_
