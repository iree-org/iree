// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model/bpe_heap.h"

void iree_tokenizer_bpe_heap_initialize(
    iree_tokenizer_bpe_heap_t* heap, iree_tokenizer_bpe_heap_entry_t* storage,
    iree_host_size_t capacity) {
  IREE_TRACE_ZONE_BEGIN(z0);
  heap->entries = storage;
  heap->capacity = capacity;
  heap->size = 0;
  IREE_TRACE_ZONE_END(z0);
}

void iree_tokenizer_bpe_heap_reset(iree_tokenizer_bpe_heap_t* heap) {
  heap->size = 0;
}

void iree_tokenizer_bpe_heap_push(iree_tokenizer_bpe_heap_t* heap,
                                  iree_tokenizer_bpe_heap_entry_t entry) {
  // Hard check: capacity is a proven bound (3 * max_token_length).
  // Exceeding it indicates a bug in the BPE algorithm, not user input.
  if (IREE_UNLIKELY(heap->size >= heap->capacity)) {
    IREE_ASSERT_UNREACHABLE("BPE heap overflow: size=%" PRIhsz
                            " capacity=%" PRIhsz
                            ". This is a bug in the BPE algorithm.",
                            heap->size, heap->capacity);
  }

  // Insert at end and sift up.
  iree_host_size_t index = heap->size++;
  heap->entries[index] = entry;

  // Sift up: swap with parent while new entry has higher priority.
  // Uses composite comparison: rank first, then start_byte for tie-breaking.
  while (index > 0) {
    iree_host_size_t parent = (index - 1) / 2;
    if (!iree_tokenizer_bpe_heap_entry_less(heap->entries[index],
                                            heap->entries[parent])) {
      break;
    }
    // Swap with parent.
    iree_tokenizer_bpe_heap_entry_t temp = heap->entries[parent];
    heap->entries[parent] = heap->entries[index];
    heap->entries[index] = temp;
    index = parent;
  }
}

iree_tokenizer_bpe_heap_entry_t iree_tokenizer_bpe_heap_pop(
    iree_tokenizer_bpe_heap_t* heap) {
  IREE_ASSERT(heap->size > 0);

  iree_tokenizer_bpe_heap_entry_t result = heap->entries[0];

  // Move last element to root and sift down.
  heap->size--;
  if (heap->size == 0) {
    return result;
  }

  heap->entries[0] = heap->entries[heap->size];

  // Sift down: swap with highest-priority child while lower priority.
  // Uses composite comparison: rank first, then start_byte for tie-breaking.
  iree_host_size_t index = 0;
  for (;;) {
    iree_host_size_t left = 2 * index + 1;
    iree_host_size_t right = 2 * index + 2;
    iree_host_size_t smallest = index;

    if (left < heap->size &&
        iree_tokenizer_bpe_heap_entry_less(heap->entries[left],
                                           heap->entries[smallest])) {
      smallest = left;
    }
    if (right < heap->size &&
        iree_tokenizer_bpe_heap_entry_less(heap->entries[right],
                                           heap->entries[smallest])) {
      smallest = right;
    }

    if (smallest == index) {
      break;
    }

    // Swap with higher-priority child.
    iree_tokenizer_bpe_heap_entry_t temp = heap->entries[index];
    heap->entries[index] = heap->entries[smallest];
    heap->entries[smallest] = temp;
    index = smallest;
  }

  return result;
}
