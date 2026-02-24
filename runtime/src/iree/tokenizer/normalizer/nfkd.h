// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_NFKD_H_
#define IREE_TOKENIZER_NORMALIZER_NFKD_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates an NFKD (Unicode Normalization Form KD) normalizer.
//
// NFKD performs compatibility decomposition followed by canonical ordering:
//   1. Compatibility decomposition (ﬁ → fi, ① → 1, ㎞ → km, etc.)
//   2. Canonical decomposition (precomposed → base + combining marks)
//   3. Canonical ordering of combining marks by CCC
//
// Unlike NFD which only performs canonical decomposition (preserving visual
// appearance), NFKD also performs compatibility decomposition which normalizes
// characters for comparison purposes (may change visual appearance).
//
// Key differences from NFD:
//   - NFD:  ﬁ stays as ﬁ (ligature preserved)
//   - NFKD: ﬁ → fi (ligature decomposed to component letters)
//
// ASCII text passes through unchanged via a fast path. Non-ASCII text is
// processed codepoint-by-codepoint, buffering combining sequences until the
// next starter (CCC=0) character arrives. The combining sequence buffer is
// bounded at 32 codepoints per Unicode Stream-Safe Text Format (UAX #15 §12);
// inputs exceeding this limit return IREE_STATUS_RESOURCE_EXHAUSTED.
//
// The decomposition buffer is 20 codepoints to handle the longest known
// compatibility decomposition (U+FDFA: Arabic ligature → 18 codepoints).
//
// Common uses of NFKD:
//   - Tokenizers requiring compatibility normalization (XLNet, ALBERT)
//   - Text search and comparison
//   - Decomposing typographic ligatures and symbols
iree_status_t iree_tokenizer_normalizer_nfkd_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_NFKD_H_
