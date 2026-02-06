// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prefix trie for O(N²) vocabulary matching in Unigram tokenization.
//
// This trie enables efficient Viterbi tokenization by allowing incremental
// prefix matching. Instead of O(N²) hash lookups (each O(L) for hashing),
// a single O(L) traversal from each starting position finds ALL matching
// vocabulary tokens.
//
// The trie is built from a token array and string table, referencing but not
// owning that data. It uses a Double-Array Aho-Corasick (DAAC) representation
// for O(1) state transitions via:
//   next_state = base[current_state] + byte
//   valid = (check[next_state] == current_state)
//
// Memory usage is approximately 12 bytes per state (base + check + output).
// For a 50K vocabulary, this typically results in ~200K states and ~2.4MB.
//
// Usage:
//   iree_tokenizer_vocab_trie_t* trie = NULL;
//   iree_tokenizer_vocab_trie_build(tokens, count, string_table,
//       allocator, &trie);
//
//   // Traverse from each position to find all matching tokens
//   iree_tokenizer_trie_cursor_t cursor;
//   iree_tokenizer_trie_cursor_reset(&cursor, trie);
//   while (iree_tokenizer_trie_cursor_advance(&cursor, input[i])) {
//     int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
//     if (token_id >= 0) { /* found a matching token */ }
//   }
//
//   iree_tokenizer_vocab_trie_free(trie);

#ifndef IREE_TOKENIZER_VOCAB_TRIE_H_
#define IREE_TOKENIZER_VOCAB_TRIE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/vocab/token.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Vocab Trie Structure
//===----------------------------------------------------------------------===//

// Double-Array Aho-Corasick trie for vocabulary prefix matching.
// State transitions are O(1): next = base[state] + byte, valid if
// check[next] == state.
typedef struct iree_tokenizer_vocab_trie_t {
  iree_allocator_t allocator;
  // Maximum token length in bytes.
  iree_host_size_t max_depth;
  iree_host_size_t node_count;  // Number of nodes (states).
  iree_host_size_t array_size;  // Size of base/check/output arrays.
  int32_t* base;                // Transition base offsets.
  // Parent state for validation (-1 = unused).
  int32_t* check;
  // Token ID at state (-1 = no token).
  int32_t* output;
} iree_tokenizer_vocab_trie_t;

// Builds an immutable prefix trie from token array and string table.
//
// The trie references but does not own |tokens| and |string_table|.
// These must remain valid for the lifetime of the trie.
//
// |tokens| is the array of token entries.
// |token_count| is the number of tokens in the array.
// |string_table| contains the UTF-8 string data referenced by tokens.
// |allocator| is used for trie storage allocation.
// |out_trie| receives the built trie on success.
iree_status_t iree_tokenizer_vocab_trie_build(
    const iree_tokenizer_token_t* tokens, iree_host_size_t token_count,
    iree_const_byte_span_t string_table, iree_allocator_t allocator,
    iree_tokenizer_vocab_trie_t** out_trie);

// Frees a trie built by iree_tokenizer_vocab_trie_build.
void iree_tokenizer_vocab_trie_free(iree_tokenizer_vocab_trie_t* trie);

// Returns the maximum token length (trie depth) in bytes.
// This is useful for bounding traversal or allocating buffers.
static inline iree_host_size_t iree_tokenizer_vocab_trie_max_depth(
    const iree_tokenizer_vocab_trie_t* trie) {
  return trie ? trie->max_depth : 0;
}

// Returns the number of nodes in the trie.
// Useful for memory analysis and debugging.
static inline iree_host_size_t iree_tokenizer_vocab_trie_node_count(
    const iree_tokenizer_vocab_trie_t* trie) {
  return trie ? trie->node_count : 0;
}

//===----------------------------------------------------------------------===//
// Cursor-based Traversal
//===----------------------------------------------------------------------===//

// Cursor for incremental trie traversal.
//
// Cursors are lightweight, stack-allocated, and require no cleanup.
// They maintain position state during byte-by-byte traversal of the trie.
//
// Example usage for finding all matching tokens from a position:
//   iree_tokenizer_trie_cursor_t cursor;
//   iree_tokenizer_trie_cursor_reset(&cursor, trie);
//   for (size_t i = 0; i < input_length; ++i) {
//     if (!iree_tokenizer_trie_cursor_advance(&cursor, input[i])) break;
//     int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
//     if (token_id >= 0) {
//       // Found token ending at position i+1
//     }
//   }
typedef struct iree_tokenizer_trie_cursor_t {
  const iree_tokenizer_vocab_trie_t* trie;
  uint32_t node_index;     // Current state (0 = root).
  iree_host_size_t depth;  // Number of bytes consumed.
} iree_tokenizer_trie_cursor_t;

// Returns the current traversal depth (bytes consumed).
static inline iree_host_size_t iree_tokenizer_trie_cursor_depth(
    const iree_tokenizer_trie_cursor_t* cursor) {
  return cursor->depth;
}

// Resets the cursor to the trie root.
// Must be called before first use or to restart traversal.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_tokenizer_trie_cursor_reset(iree_tokenizer_trie_cursor_t* cursor,
                                 const iree_tokenizer_vocab_trie_t* trie) {
  cursor->trie = trie;
  cursor->node_index = 0;  // Root state.
  cursor->depth = 0;
}

// Advances the cursor by one byte.
//
// Returns true if an edge exists for |byte| and the cursor was advanced.
// Returns false if no edge exists; cursor state is unchanged.
//
// When false is returned, no further tokens can match from this position,
// and the caller should stop traversing.
//
// This is O(1): one array lookup plus one comparison.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool
iree_tokenizer_trie_cursor_advance(iree_tokenizer_trie_cursor_t* cursor,
                                   uint8_t byte) {
  const iree_tokenizer_vocab_trie_t* trie = cursor->trie;
  if (IREE_UNLIKELY(!trie)) return false;

  int32_t current = (int32_t)cursor->node_index;
  int32_t base = trie->base[current];

  // Validate base before arithmetic.
  if (base < 0 || base > INT32_MAX - 255) {
    return false;
  }
  int32_t next_signed = base + (int32_t)byte;
  iree_host_size_t next = (iree_host_size_t)next_signed;

  // Bounds check and parent validation.
  if (next >= trie->array_size || trie->check[next] != current) {
    return false;
  }

  cursor->node_index = (uint32_t)next;
  ++cursor->depth;
  return true;
}

// Returns the token ID if current position is a word boundary, -1 otherwise.
//
// A word boundary means a vocabulary token ends exactly at this position.
// Multiple tokens may share prefixes, so this can return valid IDs at
// multiple depths during a single traversal.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE int32_t
iree_tokenizer_trie_cursor_token_id(
    const iree_tokenizer_trie_cursor_t* cursor) {
  const iree_tokenizer_vocab_trie_t* trie = cursor->trie;
  if (IREE_UNLIKELY(!trie)) return -1;
  return trie->output[cursor->node_index];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_VOCAB_TRIE_H_
