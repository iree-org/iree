// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_NFC_H_
#define IREE_TOKENIZER_NORMALIZER_NFC_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates an NFC (Unicode Normalization Form C) normalizer.
//
// NFC performs canonical decomposition followed by canonical composition:
//   1. Singleton decomposition (CJK compatibility ideographs, Hangul syllables)
//   2. Canonical ordering of combining marks by CCC (Canonical Combining Class)
//   3. Canonical composition (base + combining -> precomposed form)
//
// ASCII text passes through unchanged via a fast path. Non-ASCII text is
// processed codepoint-by-codepoint, buffering combining sequences until the
// next starter (CCC=0) character arrives. The combining sequence buffer is
// bounded at 32 codepoints per Unicode Stream-Safe Text Format (UAX #15 §12);
// inputs exceeding this limit return IREE_STATUS_RESOURCE_EXHAUSTED.
iree_status_t iree_tokenizer_normalizer_nfc_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_NFC_H_
