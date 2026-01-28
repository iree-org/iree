// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ByteLevel text transform (GPT-2 style).
//
// Maps each byte to a unique Unicode character, allowing any byte sequence to
// be represented as valid Unicode text. This is the pre-tokenization used by
// GPT-2, GPT-3, and many other models.
//
// Note: With ADD_PREFIX_SPACE, encode/decode is not perfectly invertible for
// inputs that already start with a space. Decode cannot distinguish between
// "space was added by encode" vs "input already had a space", so it always
// strips the leading space. This matches HuggingFace tokenizers and llama.cpp.

#ifndef IREE_TOKENIZER_TRANSFORMS_BYTE_LEVEL_H_
#define IREE_TOKENIZER_TRANSFORMS_BYTE_LEVEL_H_

#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/transforms/transform.h"

#ifdef __cplusplus
extern "C" {
#endif

// ByteLevel encode with inline normalization.
// Maps bytes to Unicode characters using GPT-2's byte-to-character mapping.
// Optionally prepends a space if the input doesn't start with an ASCII space.
// Normalization is applied per-codepoint as the text is scanned.
// Emits transformed text as a single segment per batch.
iree_status_t iree_tokenizer_byte_level_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_byte_level_config_t* config, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

// ByteLevel decode: maps Unicode characters back to bytes.
// Reverses the GPT-2 byte-to-character mapping. Strips leading space if
// ADD_PREFIX_SPACE flag was set during encode. Writes directly to out_buffer.
iree_status_t iree_tokenizer_byte_level_decode(
    const iree_tokenizer_byte_level_config_t* config, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TRANSFORMS_BYTE_LEVEL_H_
