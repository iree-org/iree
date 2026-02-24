// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_DEFERRED_H_
#define IREE_TOKENIZER_NORMALIZER_DEFERRED_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum bytes the deferred normalizer can buffer.
// Set larger than 256 to test exceeding sequence intermediate buffer size.
#define IREE_TOKENIZER_NORMALIZER_DEFERRED_BUFFER_SIZE 512

// Allocates a deferred normalizer that buffers all input during process()
// and only emits during finalize().
//
// This normalizer is designed to test sequence finalize behavior:
// - process(): consumes input into internal buffer, writes 0 to output
// - finalize(): emits buffered data to output
// - has_pending(): returns true if stored_count > emitted_count
//
// By deferring all output to finalize(), this normalizer exposes bugs in
// sequence finalize loops that don't handle children with more finalize
// data than the intermediate buffer can hold in a single call.
//
// For testing sequence finalize truncation bugs.
iree_status_t iree_tokenizer_normalizer_deferred_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_DEFERRED_H_
