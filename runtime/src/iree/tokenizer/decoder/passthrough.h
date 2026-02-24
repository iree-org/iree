// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Passthrough decoder for testing.
//
// This decoder concatenates token strings unchanged. It exists only for testing
// the pipeline vtable machinery. In production, a NULL decoder means "skip
// decoding entirely" - no vtable call overhead.
//
// This file should only be linked into test binaries.

#ifndef IREE_TOKENIZER_DECODER_PASSTHROUGH_H_
#define IREE_TOKENIZER_DECODER_PASSTHROUGH_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a passthrough decoder that concatenates token strings unchanged.
// For testing vtable dispatch; production uses NULL for no-op.
iree_status_t iree_tokenizer_decoder_passthrough_allocate(
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_PASSTHROUGH_H_
