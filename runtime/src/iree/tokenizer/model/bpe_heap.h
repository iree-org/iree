// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_MODEL_BPE_HEAP_H_
#define IREE_TOKENIZER_MODEL_BPE_HEAP_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// iree_tokenizer_bpe_heap_t
//===----------------------------------------------------------------------===//

// Entry in the BPE merge candidate heap storing merge rank and byte position.
// The heap orders entries by rank (lower rank = higher priority = applied
// first). For equal ranks, the entry with smaller start_byte has higher
// priority - this ensures leftmost merges are applied first, which is critical
// for correctness when overlapping merges have the same rank.
//
// The left_start_byte field identifies the left token's starting byte
// position in the segment, which is stable even as window indices shift during
// merges.
//
// Entries may become stale if the tokens at their position have been merged
// or removed. The BPE algorithm uses lazy invalidation: stale entries are
// detected and discarded when popped rather than eagerly removed when merges
// occur. This avoids the O(n) cost of finding and removing entries.
//
// Using byte position instead of window index is critical for correctness:
// when a merge removes a token from the window, all subsequent window indices
// shift down, invalidating any heap entries that stored those indices. Byte
// positions are stable because they refer to the input segment, not the window.
typedef struct iree_tokenizer_bpe_heap_entry_t {
  uint32_t rank;  // Merge priority (0 = highest).
  // Starting byte of left token in segment.
  uint32_t left_start_byte;
} iree_tokenizer_bpe_heap_entry_t;

// Returns true if entry |a| has higher priority (should be closer to root)
// than entry |b|. Priority is determined by rank first, then by start_byte
// for equal ranks (smaller start_byte = higher priority = leftmost first).
static inline bool iree_tokenizer_bpe_heap_entry_less(
    iree_tokenizer_bpe_heap_entry_t a, iree_tokenizer_bpe_heap_entry_t b) {
  if (a.rank != b.rank) return a.rank < b.rank;
  return a.left_start_byte < b.left_start_byte;
}

// Min-heap for BPE merge candidates ordered by rank.
//
// This is a standard binary min-heap stored in an array where the parent of
// element at index i is at (i-1)/2 and children are at 2i+1 and 2i+2. The
// minimum element (lowest rank = highest priority merge) is always at index 0.
//
// The heap uses pre-allocated storage provided at initialization time. This
// allows the BPE state to allocate the heap buffer as part of a single slab
// allocation, avoiding per-operation allocations during encoding.
//
// Capacity is 3*max_token_length, derived from rigorous analysis:
// - Peak heap = H + (W - 1) where H = entries at call, W = window tokens
// - H ≤ L - 1 (bytes added between heap drains), W ≤ 2L - 1 (window bound)
// - Peak ≤ (L - 1) + (2L - 2) = 3L - 3 < 3L
// Exceeding capacity indicates a bug in the algorithm.
typedef struct iree_tokenizer_bpe_heap_t {
  iree_tokenizer_bpe_heap_entry_t* entries;
  iree_host_size_t capacity;
  iree_host_size_t size;
} iree_tokenizer_bpe_heap_t;

// Initializes a heap with pre-allocated storage.
// The storage array must have space for at least |capacity| entries and must
// remain valid for the lifetime of the heap.
void iree_tokenizer_bpe_heap_initialize(
    iree_tokenizer_bpe_heap_t* heap, iree_tokenizer_bpe_heap_entry_t* storage,
    iree_host_size_t capacity);

// Resets the heap to empty without deallocating storage.
// This is O(1) and simply sets the size to zero.
void iree_tokenizer_bpe_heap_reset(iree_tokenizer_bpe_heap_t* heap);

// Returns true if the heap contains no entries.
static inline bool iree_tokenizer_bpe_heap_is_empty(
    const iree_tokenizer_bpe_heap_t* heap) {
  return heap->size == 0;
}

// Returns the number of entries in the heap.
static inline iree_host_size_t iree_tokenizer_bpe_heap_size(
    const iree_tokenizer_bpe_heap_t* heap) {
  return heap->size;
}

// Pushes an entry onto the heap.
// The entry is inserted and sifted up to maintain heap ordering.
// Requires: heap->size < heap->capacity.
void iree_tokenizer_bpe_heap_push(iree_tokenizer_bpe_heap_t* heap,
                                  iree_tokenizer_bpe_heap_entry_t entry);

// Returns the minimum entry without removing it.
// Requires: !iree_tokenizer_bpe_heap_is_empty(heap).
static inline iree_tokenizer_bpe_heap_entry_t iree_tokenizer_bpe_heap_peek(
    const iree_tokenizer_bpe_heap_t* heap) {
  return heap->entries[0];
}

// Removes and returns the minimum entry from the heap.
// The last entry is moved to the root and sifted down to restore ordering.
// Requires: !iree_tokenizer_bpe_heap_is_empty(heap).
iree_tokenizer_bpe_heap_entry_t iree_tokenizer_bpe_heap_pop(
    iree_tokenizer_bpe_heap_t* heap);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_MODEL_BPE_HEAP_H_
