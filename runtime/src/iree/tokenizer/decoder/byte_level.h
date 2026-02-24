// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_DECODER_BYTE_LEVEL_H_
#define IREE_TOKENIZER_DECODER_BYTE_LEVEL_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a ByteLevel decoder that reverses GPT-2's byte-to-unicode mapping.
//
// GPT-2/RoBERTa tokenizers encode arbitrary bytes as Unicode codepoints:
// - Shifted range (0x100-0x143): Control chars, space, DEL, C1, NBSP, soft
// hyphen
// - Identity range (0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF): Map to themselves
// - Passthrough: Codepoints not in the 256-byte mapping (emoji, CJK, etc.)
//
// The decoder:
// 1. Maps shifted/identity codepoints to their original byte values
// 2. Passes through non-mapping codepoints unchanged
// 3. Validates the resulting bytes as UTF-8 (lossy, invalid -> U+FFFD)
//
// This is a SHRINKING decoder: identity/shifted codepoints (2 bytes UTF-8)
// become single bytes, while passthrough stays the same size.
iree_status_t iree_tokenizer_decoder_byte_level_allocate(
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_BYTE_LEVEL_H_
