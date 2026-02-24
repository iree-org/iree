// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_REGEX_REPLACE_H_
#define IREE_TOKENIZER_NORMALIZER_REGEX_REPLACE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum replacement content length for regex replace normalizer.
// Empirical max in production tokenizers: 1 byte (single space).
// 32 bytes provides margin while keeping state compact.
#define IREE_TOKENIZER_REGEX_REPLACE_MAX_CONTENT 32

// Allocates a regex-based replace normalizer that substitutes all matches of
// the given regex pattern with the specified content.
//
// Uses the tokenizer regex engine with its built-in 256-byte rewind buffer
// for cross-chunk backtracking. Implements leftmost-longest matching semantics
// matching HuggingFace tokenizers behavior.
//
// Production patterns (from common tokenizers):
//   - ` {2,}` -> ` `: DeBERTa-v3, BGE-M3, mDeBERTa-v3 (collapse multiple
//   spaces)
//   - `\s+` -> ` `: CLIP (normalize all whitespace to single space)
//
// |pattern|: the regex pattern to match (must be non-empty, uses tokenizer
//            regex syntax - see iree/tokenizer/regex/compile.h).
// |content|: the byte sequence to substitute (0-32 bytes, empty = deletion).
//
// Returns:
//   IREE_STATUS_OK on success.
//   IREE_STATUS_INVALID_ARGUMENT if pattern is empty, content too long, or
//       pattern contains invalid regex syntax.
//   IREE_STATUS_FAILED_PRECONDITION if pattern uses unsupported regex features.
//   IREE_STATUS_RESOURCE_EXHAUSTED if regex compilation exceeds state limits.
iree_status_t iree_tokenizer_normalizer_regex_replace_allocate(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_REGEX_REPLACE_H_
