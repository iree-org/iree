// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_VOCAB_MERGE_HASH_H_
#define IREE_TOKENIZER_VOCAB_MERGE_HASH_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

struct iree_tokenizer_vocab_t;

//===----------------------------------------------------------------------===//
// iree_tokenizer_vocab_merge_hash_t
//===----------------------------------------------------------------------===//

// Result of a merge hash lookup containing the merge rank and result token.
typedef struct iree_tokenizer_merge_hash_result_t {
  // Merge priority where 0 is highest priority (applied first).
  // Set to UINT32_MAX if no merge exists for the given pair.
  uint32_t rank;
  // Token ID produced by merging the left and right tokens.
  // Only valid when rank != UINT32_MAX.
  int32_t result_id;
} iree_tokenizer_merge_hash_result_t;

// Returns a result indicating no merge exists for a token pair.
static inline iree_tokenizer_merge_hash_result_t
iree_tokenizer_merge_hash_result_none(void) {
  iree_tokenizer_merge_hash_result_t result = {UINT32_MAX, -1};
  return result;
}

// Returns true if the result indicates a valid merge exists.
static inline bool iree_tokenizer_merge_hash_result_is_valid(
    iree_tokenizer_merge_hash_result_t result) {
  return result.rank != UINT32_MAX;
}

// Sentinel value for empty hash slots.
// Using UINT64_MAX instead of 0 because key=0 is valid (merge of token 0 + 0).
#define IREE_TOKENIZER_MERGE_HASH_EMPTY_KEY UINT64_MAX

// Hash table entry storing the packed key and lookup result.
typedef struct iree_tokenizer_merge_hash_entry_t {
  // (left_id << 32) | right_id, EMPTY_KEY for empty slots.
  uint64_t key;
  uint32_t rank;
  int32_t result_id;
} iree_tokenizer_merge_hash_entry_t;

// Hash table for O(1) BPE merge lookup by token pair.
//
// BPE encoding requires repeatedly finding the highest-priority merge that can
// be applied to adjacent tokens. The naive approach of scanning all merge rules
// for each pair is O(M) per lookup where M is the number of merges (typically
// 50k-100k). This hash table provides O(1) lookup by using the packed token
// pair (left_id << 32 | right_id) as the key.
//
// The table uses open addressing with linear probing and a load factor of ~0.7.
// Empty slots are marked with key=UINT64_MAX sentinel.
//
// This structure is built once during BPE model initialization from the vocab's
// merge list and is immutable thereafter. It is owned by the BPE model, not the
// vocab, keeping BPE-specific data structures out of the core vocab.
//
// Uses a flexible array member (FAM) for entries to enable direct indexing
// without pointer indirection in the lookup hot path.
typedef struct iree_tokenizer_vocab_merge_hash_t {
  iree_allocator_t allocator;
  iree_host_size_t capacity;  // Power of 2.
  iree_host_size_t count;
  iree_tokenizer_merge_hash_entry_t entries[];
} iree_tokenizer_vocab_merge_hash_t;

// Builds a merge hash table from a vocabulary's merge rules.
// The vocab's merges are stored in rank order where index 0 is the highest
// priority merge. The result token for each merge is looked up from the vocab.
iree_status_t iree_tokenizer_vocab_merge_hash_build(
    const struct iree_tokenizer_vocab_t* vocab, iree_allocator_t allocator,
    iree_tokenizer_vocab_merge_hash_t** out_hash);

// Packs two token IDs into a single 64-bit key for hash lookup.
static inline uint64_t iree_tokenizer_merge_hash_pack_key(int32_t left_id,
                                                          int32_t right_id) {
  return ((uint64_t)(uint32_t)left_id << 32) | (uint64_t)(uint32_t)right_id;
}

// Murmur3 64-bit finalizer: excellent mixing for packed integer keys.
// Our keys are (left_id << 32) | right_id where IDs are typically 0-50000.
// The XOR-shift-multiply pattern mixes all 64 bits thoroughly.
static inline iree_host_size_t iree_tokenizer_merge_hash_fn(uint64_t key) {
  uint64_t x = key;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return (iree_host_size_t)x;
}

// Looks up a merge by token pair, returning the rank and result token.
// Returns a result with rank=UINT32_MAX if no merge exists for the pair.
// Inlined for performance - called frequently in BPE pair validation.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE iree_tokenizer_merge_hash_result_t
iree_tokenizer_vocab_merge_hash_lookup(
    const iree_tokenizer_vocab_merge_hash_t* hash, int32_t left_token_id,
    int32_t right_token_id) {
  if (!hash || hash->capacity == 0) {
    return iree_tokenizer_merge_hash_result_none();
  }

  uint64_t key =
      iree_tokenizer_merge_hash_pack_key(left_token_id, right_token_id);
  iree_host_size_t mask = hash->capacity - 1;
  iree_host_size_t index = iree_tokenizer_merge_hash_fn(key) & mask;

  // Linear probe until we find the key or an empty slot.
  while (hash->entries[index].key != IREE_TOKENIZER_MERGE_HASH_EMPTY_KEY) {
    if (hash->entries[index].key == key) {
      iree_tokenizer_merge_hash_result_t result = {
          hash->entries[index].rank,
          hash->entries[index].result_id,
      };
      return result;
    }
    index = (index + 1) & mask;
  }

  return iree_tokenizer_merge_hash_result_none();
}

// Frees a merge hash table and all associated memory.
void iree_tokenizer_vocab_merge_hash_free(
    iree_tokenizer_vocab_merge_hash_t* hash);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_VOCAB_MERGE_HASH_H_
