// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_DECODER_WORDPIECE_H_
#define IREE_TOKENIZER_DECODER_WORDPIECE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum prefix length supported by WordPiece decoder.
// Empirical analysis of HuggingFace tokenizers shows all WordPiece decoders
// use: prefix="##" (2 bytes). 8 bytes provides headroom for other scripts.
#define IREE_TOKENIZER_WORDPIECE_MAX_PREFIX_LENGTH 8

// Configuration for WordPiece decoder.
typedef struct iree_tokenizer_decoder_wordpiece_config_t {
  // Prefix marking continuation subwords (default "##").
  iree_string_view_t prefix;
  // Whether to apply cleanup transformations for punctuation/contractions.
  bool cleanup;
} iree_tokenizer_decoder_wordpiece_config_t;

// Creates a WordPiece configuration with the given continuation prefix and
// cleanup flag.
static inline iree_tokenizer_decoder_wordpiece_config_t
iree_tokenizer_make_decoder_wordpiece_config(iree_string_view_t prefix,
                                             bool cleanup) {
  iree_tokenizer_decoder_wordpiece_config_t config = {/*prefix=*/prefix,
                                                      /*cleanup=*/cleanup};
  return config;
}

// Allocates a WordPiece decoder for BERT-style tokenizers.
//
// The decoder handles subword tokens:
// - Tokens starting with |prefix| (typically "##") have the prefix stripped
//   and are joined directly to the previous token.
// - Other tokens (except the first) have a space prepended.
//
// If |cleanup| is true, applies transformations to fix tokenization artifacts:
// - Removes space before punctuation: " ." -> "."
// - Fixes contractions: " n't" -> "n't", " 'm" -> "'m", etc.
//
// This decoder can expand output (space prepending adds bytes). Callers
// must provide sufficient output buffer capacity.
iree_status_t iree_tokenizer_decoder_wordpiece_allocate(
    iree_tokenizer_decoder_wordpiece_config_t config,
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_WORDPIECE_H_
