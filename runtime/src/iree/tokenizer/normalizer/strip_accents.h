// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS_H_
#define IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a strip_accents normalizer that removes Unicode combining marks.
//
// This normalizer does not perform NFD decomposition. It only filters
// out characters in the Unicode Mark category (Mn, Mc, Me).
// Precomposed characters like 'e' (U+00E9) pass through unchanged because
// they are single codepoints, not base + combining mark sequences.
//
// To strip accents from precomposed characters, chain an NFD normalizer
// before this one: NFD -> StripAccents. This matches HuggingFace behavior.
//
// Example:
//   Input "cafe" (composed e):     Output "cafe" (unchanged)
//   Input "cafe\xCC\x81" (e+mark): Output "cafe" (mark removed)
//
// Mark categories filtered:
//   - Mn (Nonspacing Mark): Combining accents, diacritics
//   - Mc (Spacing Combining Mark): Vowel signs in Indic scripts
//   - Me (Enclosing Mark): Circles, squares around characters
iree_status_t iree_tokenizer_normalizer_strip_accents_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS_H_
