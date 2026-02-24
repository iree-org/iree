// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_PREPEND_H_
#define IREE_TOKENIZER_NORMALIZER_PREPEND_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum supported prepend string length in bytes.
// This bounds the prefix comparison buffer and state storage. The value is
// chosen to accommodate typical prepend strings (e.g., "▁" = 3 bytes) with
// substantial margin while keeping state compact for embedding.
// Empirical max in production tokenizers: 3 bytes ("▁").
#define IREE_TOKENIZER_NORMALIZER_PREPEND_MAX_LENGTH 16

// Allocates a prepend normalizer that emits a fixed string before input.
//
// The prepend string is emitted before the first byte of non-empty input.
// Empty input produces empty output (no prepend). The prepend string is
// copied internally and does not need to remain valid after this call.
//
// When |skip_if_prefix_matches| is true, the normalizer only prepends if the
// input does not already start with the prepend string. This implements the
// HuggingFace Metaspace prepend_scheme="first" behavior where spaces are
// replaced with ▁ upstream and ▁ is only prepended to text that doesn't
// already begin with it.
//
// The prepend string must be IREE_TOKENIZER_NORMALIZER_PREPEND_MAX_LENGTH
// bytes or fewer.
iree_status_t iree_tokenizer_normalizer_prepend_allocate(
    iree_string_view_t prepend_string, bool skip_if_prefix_matches,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_PREPEND_H_
