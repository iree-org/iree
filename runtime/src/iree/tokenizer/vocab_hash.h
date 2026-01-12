// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable hash table for O(1) vocabulary token lookup (string -> token ID).
//
// This hash table is built once from a token array and string table, then
// supports fast lookups. Uses CRC32C hashing (hardware-accelerated on x86_64)
// with linear probing for cache-efficient collision resolution.

#ifndef IREE_TOKENIZER_VOCAB_HASH_H_
#define IREE_TOKENIZER_VOCAB_HASH_H_

#include "iree/base/api.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Vocab Hash Table
//===----------------------------------------------------------------------===//

// Default load factor for hash table sizing (75%).
// Lower values use more memory but reduce probe chain length.
// Typical values: 75 (compact), 50 (faster lookups).
#define IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT 75

// Opaque hash table type for vocabulary lookup.
typedef struct iree_tokenizer_vocab_hash_t iree_tokenizer_vocab_hash_t;

// Builds an immutable hash table from token array and string table.
//
// The hash table references but does not own |tokens| and |string_table|.
// These must remain valid for the lifetime of the hash table.
//
// |tokens| is the array of token entries.
// |token_count| is the number of tokens in the array.
// |string_table| contains the UTF-8 string data referenced by tokens.
// |load_percent| is the target load factor (1-100). Use
//     IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT for typical usage.
// |allocator| is used for hash table storage allocation.
// |out_hash| receives the built hash table on success.
iree_status_t iree_tokenizer_vocab_hash_build(
    const iree_tokenizer_token_t* tokens, iree_host_size_t token_count,
    iree_const_byte_span_t string_table, uint8_t load_percent,
    iree_allocator_t allocator, iree_tokenizer_vocab_hash_t** out_hash);

// Looks up a string in the hash table.
// Returns the token ID if found, or -1 if not found.
int32_t iree_tokenizer_vocab_hash_lookup(
    const iree_tokenizer_vocab_hash_t* hash, iree_string_view_t text);

// Frees a hash table built by iree_tokenizer_vocab_hash_build.
void iree_tokenizer_vocab_hash_free(iree_tokenizer_vocab_hash_t* hash);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_VOCAB_HASH_H_
