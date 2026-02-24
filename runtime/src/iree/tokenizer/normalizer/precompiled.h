// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_PRECOMPILED_H_
#define IREE_TOKENIZER_NORMALIZER_PRECOMPILED_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a Precompiled (SentencePiece) normalizer from binary charsmap data.
//
// The precompiled normalizer uses a double-array trie for fast prefix matching
// of Unicode sequences to normalized replacement strings. This implements NFKC
// and similar normalizations as used by SentencePiece-based tokenizers.
//
// ALGORITHM (matching HuggingFace tokenizers):
//   For each grapheme in input:
//     1. If grapheme.length < 6 bytes:
//          Try trie lookup on whole grapheme
//          If match found: emit replacement, advance past grapheme, continue
//     2. Character-by-character fallback:
//          For each codepoint in grapheme:
//            Perform longest-match trie lookup starting at codepoint
//            If match: emit replacement, advance by matched bytes
//            Else: emit original codepoint bytes (passthrough)
//
// BINARY FORMAT (precompiled_charsmap after base64 decoding):
//   [4 bytes: trie_size (little-endian uint32, size in bytes)]
//   [trie_size bytes: double-array trie units (uint32[])]
//   [remaining bytes: NUL-terminated replacement string pool]
//
// INTERNAL POOL FORMAT:
//   During allocation, the NUL-terminated pool is transformed to
//   length-prefixed format for O(1) string lookup. Pool offset translation:
//   stored = input + 1.
//
// STREAMING:
//   Uses a 32-byte overlap buffer for pattern matching across chunk boundaries
//   and a 64-byte pending buffer for replacement strings awaiting output.
//   Both buffers are sized to guarantee identical behavior regardless of
//   chunk boundaries.
//
// Parameters:
//   charsmap: Raw binary blob (after base64 decoding from JSON).
//   allocator: Allocator for the normalizer and its internal data.
//   out_normalizer: Receives the allocated normalizer on success.
//
// Returns IREE_STATUS_INVALID_ARGUMENT if the format is invalid.
iree_status_t iree_tokenizer_precompiled_normalizer_allocate(
    iree_const_byte_span_t charsmap, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_PRECOMPILED_H_
