// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_PASSTHROUGH_H_
#define IREE_TOKENIZER_NORMALIZER_PASSTHROUGH_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a passthrough normalizer that copies input to output unchanged.
//
// This normalizer passes input through unchanged. It exists only for testing
// the pipeline vtable machinery. In production, a NULL normalizer means "skip
// normalization entirely" - no vtable call overhead.
//
// For testing vtable dispatch; production uses NULL for no-op.
iree_status_t iree_tokenizer_normalizer_passthrough_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_PASSTHROUGH_H_
