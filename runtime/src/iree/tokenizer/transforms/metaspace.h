// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Metaspace text transform (SentencePiece-style).
//
// Replaces spaces with a replacement character (default: U+2581 Lower One
// Eighth Block). Supports configurable prepend schemes and optional splitting.
//
// Note: With PREPEND_ALWAYS or PREPEND_FIRST, encode/decode is not perfectly
// invertible for inputs that already start with whitespace. Decode cannot
// distinguish between "replacement was prepended by encode" vs "input already
// had leading whitespace", so it always strips the leading replacement. This
// matches HuggingFace tokenizers and SentencePiece behavior.

#ifndef IREE_TOKENIZER_TRANSFORMS_METASPACE_H_
#define IREE_TOKENIZER_TRANSFORMS_METASPACE_H_

#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/transforms/transform.h"

#ifdef __cplusplus
extern "C" {
#endif

// Metaspace encode with inline normalization.
// Replaces spaces with replacement character.
// Optionally prepends the replacement character and/or splits on it.
// Normalization is applied per-codepoint as the text is scanned.
// Emits segments via callback (batched for efficiency).
iree_status_t iree_tokenizer_metaspace_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_metaspace_config_t* config, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

// Metaspace decode: replaces replacement character with spaces.
// Strips leading replacement if prepend_scheme is ALWAYS or FIRST.
// Writes directly to out_buffer.
iree_status_t iree_tokenizer_metaspace_decode(
    const iree_tokenizer_metaspace_config_t* config, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TRANSFORMS_METASPACE_H_
