// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_NFD_H_
#define IREE_TOKENIZER_NORMALIZER_NFD_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates an NFD (Unicode Normalization Form D) normalizer.
//
// NFD performs canonical decomposition without composition:
//   1. Full canonical decomposition (precomposed -> base + combining marks)
//   2. Canonical ordering of combining marks by CCC (Canonical Combining Class)
//
// Unlike NFC, NFD does not recompose the decomposed characters. This means
// precomposed characters like 'é' (U+00E9) become 'e' + combining acute
// (U+0301).
//
// ASCII text passes through unchanged via a fast path. Non-ASCII text is
// processed codepoint-by-codepoint, buffering combining sequences until the
// next starter (CCC=0) character arrives. The combining sequence buffer is
// bounded at 32 codepoints per Unicode Stream-Safe Text Format (UAX #15 §12);
// inputs exceeding this limit return IREE_STATUS_RESOURCE_EXHAUSTED.
//
// Common uses of NFD:
//   - Preprocessing for accent stripping (NFD + StripAccents)
//   - Unicode text normalization for comparison
//   - Decomposing Hangul syllables to Jamo
iree_status_t iree_tokenizer_normalizer_nfd_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_NFD_H_
