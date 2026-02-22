// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_DECODER_BYTE_FALLBACK_H_
#define IREE_TOKENIZER_DECODER_BYTE_FALLBACK_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses a token matching the `<0xHH>` pattern (where HH is a hex byte).
// Returns true if the token matches, with the byte value in |out_byte|.
bool iree_tokenizer_decoder_byte_fallback_parse_byte_token(
    iree_string_view_t token, uint8_t* out_byte);

// Allocates a ByteFallback decoder for byte-level tokenizers.
//
// Transforms tokens matching the pattern `<0xHH>` (where HH is a hex byte)
// back into raw bytes, validating UTF-8 sequences. Non-matching tokens pass
// through unchanged.
//
// Example token sequence: ["Hello", "<0xC3>", "<0xA9>", "!"]
//   -> "Helloé!" (where é is U+00E9, UTF-8 encoded as C3 A9)
//
// UTF-8 validation:
// - Valid sequences (1-4 bytes) are emitted as-is
// - Invalid bytes or malformed sequences emit U+FFFD per bad byte
// - Incomplete sequences at stream end emit U+FFFD per remaining byte
iree_status_t iree_tokenizer_decoder_byte_fallback_allocate(
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_BYTE_FALLBACK_H_
