// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for vocab structure definition.
// Used by vocab.c and vocab_builder.c.

#ifndef IREE_TOKENIZER_VOCAB_INTERNAL_H_
#define IREE_TOKENIZER_VOCAB_INTERNAL_H_

#include "iree/base/api.h"
#include "iree/tokenizer/types.h"
#include "iree/tokenizer/vocab_hash.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal vocabulary structure.
struct iree_tokenizer_vocab_t {
  iree_allocator_t allocator;

  // Token data (owned).
  // Arrays are sized (max_token_id + 1) to allow O(1) lookup by ID.
  // Gap slots (missing IDs in sparse vocabs) have ATTR_UNUSED set.
  iree_tokenizer_token_t* tokens;
  float* scores;
  iree_host_size_t token_count;  // Actual number of tokens (excludes gaps).
  int32_t max_token_id;          // Highest valid token ID (-1 if empty).

  // String table (owned).
  uint8_t* string_data;
  iree_host_size_t string_size;

  // Hash table for lookup (owned).
  iree_tokenizer_vocab_hash_t* hash;

  // Special token IDs.
  iree_tokenizer_special_ids_t special_ids;

  // BPE merge rules (owned, NULL for non-BPE vocabs).
  // Stored in rank order: index 0 = highest priority merge.
  iree_tokenizer_merge_t* merges;
  iree_host_size_t merge_count;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_VOCAB_INTERNAL_H_
