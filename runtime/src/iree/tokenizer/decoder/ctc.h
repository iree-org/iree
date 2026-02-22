// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTC (Connectionist Temporal Classification) decoder for speech models.
//
// Post-processes CTC model output (Wav2Vec2, HuBERT, etc.) by:
// - Deduplicating consecutive identical tokens (CTC alignment artifacts)
// - Removing pad/blank tokens (CTC's silence marker)
// - Converting word delimiter tokens to spaces
// - Applying optional wordpiece cleanup (punctuation spacing, contractions)
//
// Example: CTC output for "hello"
//   Input:  ["<pad>", "<pad>", "h", "e", "e", "l", "l", "<pad>", "l", "o", "o"]
//   Output: "hello"
//
// Pad tokens act as dedup breakers - two "l"s separated by pad produce "ll".
//
// Algorithm matches HuggingFace tokenizers CTC decoder:
// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/ctc.rs

#ifndef IREE_TOKENIZER_DECODER_CTC_H_
#define IREE_TOKENIZER_DECODER_CTC_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum size for pad_token and word_delimiter_token strings.
// Also limits per-token size for dedup comparison and output buffering.
#define IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE 64

// Allocates a CTC decoder with custom configuration.
// |pad_token| is the blank/padding token to remove (e.g., "<pad>", "[PAD]").
// |word_delimiter_token| is converted to space when cleanup=true (e.g., "|").
// |cleanup| enables wordpiece cleanup rules and word delimiter replacement.
// When false, tokens pass through with only dedup and pad removal.
// Token strings are copied; the caller need not keep them alive.
// Returns IREE_STATUS_INVALID_ARGUMENT if tokens exceed max size.
iree_status_t iree_tokenizer_decoder_ctc_allocate(
    iree_string_view_t pad_token, iree_string_view_t word_delimiter_token,
    bool cleanup, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_CTC_H_
