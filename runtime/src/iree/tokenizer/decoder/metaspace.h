// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_DECODER_METASPACE_H_
#define IREE_TOKENIZER_DECODER_METASPACE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepend scheme for Metaspace decoder.
typedef enum iree_tokenizer_decoder_metaspace_prepend_scheme_e {
  // Always strip leading metaspace from first token.
  IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS = 0,
  // Never strip leading metaspace.
  IREE_TOKENIZER_DECODER_METASPACE_PREPEND_NEVER = 1,
  // Only strip if first token starts with metaspace (first word behavior).
  IREE_TOKENIZER_DECODER_METASPACE_PREPEND_FIRST = 2,
} iree_tokenizer_decoder_metaspace_prepend_scheme_t;

// Allocates a Metaspace decoder.
//
// |replacement_codepoint| is the metaspace character to replace with space.
//   Use 0 for default (U+2581, LOWER ONE EIGHTH BLOCK).
// |prepend_scheme| controls leading metaspace stripping behavior.
iree_status_t iree_tokenizer_decoder_metaspace_allocate(
    uint32_t replacement_codepoint,
    iree_tokenizer_decoder_metaspace_prepend_scheme_t prepend_scheme,
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_METASPACE_H_
