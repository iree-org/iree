// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_REPLACE_H_
#define IREE_TOKENIZER_NORMALIZER_REPLACE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum pattern length for replace normalizer.
// Empirical max in production tokenizers: 3 bytes ("▁").
// 32 bytes provides 10x margin while keeping state compact.
#define IREE_TOKENIZER_REPLACE_MAX_PATTERN 32

// Maximum replacement content length for replace normalizer.
// Empirical max in production tokenizers: 3 bytes.
#define IREE_TOKENIZER_REPLACE_MAX_CONTENT 32

// Allocates a replace normalizer that substitutes all occurrences of a literal
// pattern with the specified content.
//
// The implementation auto-selects between two optimized codepaths:
//
// Single-byte patterns (pattern.size == 1):
//   Zero-buffering streaming replacement. No internal state beyond tracking
//   partial content emission when output buffer is smaller than content.
//
// Multi-byte patterns (pattern.size > 1):
//   Uses a small overlap buffer (pattern_length - 1 bytes) to handle pattern
//   matches that span chunk boundaries. Employs memchr + memcmp for SIMD-
//   accelerated scanning.
//
// Both paths emit replacement content directly when output capacity permits,
// falling back to a pending buffer only when output is exhausted mid-emission
// (rare in practice; optimized for the 98%+ case where this doesn't happen).
//
// |pattern|: the byte sequence to match (1-32 bytes, must be non-empty).
// |content|: the byte sequence to substitute (0-32 bytes, empty = deletion).
//
// Returns IREE_STATUS_INVALID_ARGUMENT if pattern is empty or exceeds limits.
iree_status_t iree_tokenizer_normalizer_replace_allocate(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_REPLACE_H_
