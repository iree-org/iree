// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Passthrough segmenter for testing.
//
// This segmenter treats each input chunk as a single segment (no splitting).
// It exists only for testing the pipeline vtable machinery. In production,
// a NULL segmenter means "treat entire input as one segment" - no vtable
// call overhead.
//
// This file should only be linked into test binaries.

#ifndef IREE_TOKENIZER_SEGMENTER_PASSTHROUGH_H_
#define IREE_TOKENIZER_SEGMENTER_PASSTHROUGH_H_

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates a passthrough segmenter that emits one segment per process() call.
// For testing vtable dispatch; production uses NULL for no-op.
iree_status_t iree_tokenizer_segmenter_passthrough_allocate(
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_PASSTHROUGH_H_
