// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Split text transform using regex pattern matching.
//
// The Split transform uses a compiled DFA to find pattern matches in the input
// text and splits the text into segments based on those matches. This is the
// pre-tokenization used by GPT-2, Llama 3, Qwen2, and other modern tokenizers.
//
// Example: With pattern "\s+" (whitespace) and behavior REMOVED:
//   Input: "hello world"
//   Output: ["hello", "world"]
//
// Example: With pattern "\s+" and behavior ISOLATED:
//   Input: "hello world"
//   Output: ["hello", " ", "world"]
//
// Example: With GPT-2 pattern and invert=true:
//   The pattern matches word tokens, so invert emits the matches themselves.

#ifndef IREE_TOKENIZER_TRANSFORMS_SPLIT_H_
#define IREE_TOKENIZER_TRANSFORMS_SPLIT_H_

#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/regex/exec.h"
#include "iree/tokenizer/transforms/transform.h"

#ifdef __cplusplus
extern "C" {
#endif

// Deinitializes a Split config, freeing owned resources.
// The config struct itself is not freed (caller manages its storage).
void iree_tokenizer_split_config_deinitialize(
    iree_tokenizer_split_config_t* config);

// Split encode with inline normalization.
//
// Finds all matches of the pattern in the input text and emits segments
// according to the configured behavior. Normalization is applied inline
// before splitting.
//
// For invert=false (default): Pattern matches are the delimiters.
// For invert=true: Pattern matches are kept as segments, non-matches discarded.
//
// |normalizer| is applied inline during scanning (may be NULL).
// |config| is the Split configuration.
// |text| is the input text to split.
// |callback| receives batches of output segments.
// |user_data| is passed to the callback.
iree_status_t iree_tokenizer_split_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_split_config_t* config, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

// Split decode: passthrough (no transformation).
// Since Split only reorganizes segments during encode, decode is identity.
iree_status_t iree_tokenizer_split_decode(
    const iree_tokenizer_split_config_t* config, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TRANSFORMS_SPLIT_H_
