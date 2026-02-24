// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Window and heap operations for the O(n log L) sliding window BPE algorithm.
//
// The sliding window maintains a bounded set of tokens (at most 2*L where L
// is the maximum token length). Tokens whose byte range ends before the
// "freeze point" (current position - L + 1) cannot be affected by future
// input and can be safely emitted.
//
// The min-heap tracks merge candidates ordered by rank. Stale entries (where
// tokens have already merged) are lazily invalidated on pop.

#include "iree/tokenizer/model/bpe_internal.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

// Linear vs binary search threshold for window find operations.
// With 32 or fewer tokens, linear scan's cache friendliness and branch
// prediction outweigh binary search's O(log n) advantage. Based on profiling
// typical workloads where windows are usually < 64 tokens.
#define IREE_TOKENIZER_BPE_WINDOW_SEARCH_LINEAR_THRESHOLD 32

//===----------------------------------------------------------------------===//
// Window Operations
//===----------------------------------------------------------------------===//

void iree_tokenizer_bpe_window_push(iree_tokenizer_bpe_state_t* state,
                                    const iree_tokenizer_bpe_model_t* model,
                                    iree_tokenizer_bpe_window_token_t token) {
  IREE_ASSERT(state->window.count < model->window_capacity);
  iree_tokenizer_bpe_window_token_t* window =
      iree_tokenizer_bpe_state_window(state, model);
  iree_host_size_t index =
      (state->window.start + state->window.count) & model->window_capacity_mask;
  window[index] = token;
  state->window.count++;
}

iree_tokenizer_bpe_window_token_t iree_tokenizer_bpe_window_pop_front(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  IREE_ASSERT(state->window.count > 0);
  iree_tokenizer_bpe_window_token_t* window =
      iree_tokenizer_bpe_state_window(state, model);
  iree_tokenizer_bpe_window_token_t token = window[state->window.start];
  state->window.start = (state->window.start + 1) & model->window_capacity_mask;
  state->window.count--;
  return token;
}

// Removes the token at logical_index by shifting later tokens down.
static void iree_tokenizer_bpe_window_remove(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model,
    iree_host_size_t logical_index) {
  IREE_ASSERT(logical_index < state->window.count);
  for (iree_host_size_t i = logical_index; i < state->window.count - 1; ++i) {
    *iree_tokenizer_bpe_window_at(state, model, i) =
        *iree_tokenizer_bpe_window_at(state, model, i + 1);
  }
  state->window.count--;
}

// Finds the window index of the token starting at the given byte position.
// Returns the logical index, or SIZE_MAX if no such token exists.
//
// Window tokens are always sorted by start_byte (appended left-to-right,
// merges preserve order). Uses tiered approach:
// - Small windows (≤32): linear scan with cache-friendly sequential access
// - Large windows (>32): binary search for O(log W) instead of O(W)
//
// The threshold of 32 balances linear scan's cache friendliness against
// binary search's algorithmic advantage. Profiling shows typical windows
// are small enough that linear scan wins due to branch prediction and
// cache behavior.
static iree_host_size_t iree_tokenizer_bpe_window_find_by_start_byte(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model,
    uint32_t start_byte) {
  iree_host_size_t count = state->window.count;

  // Small windows: linear scan is faster due to cache locality and
  // predictable branches.
  if (IREE_LIKELY(count <= IREE_TOKENIZER_BPE_WINDOW_SEARCH_LINEAR_THRESHOLD)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_tokenizer_bpe_window_token_t* token =
          iree_tokenizer_bpe_window_at(state, model, i);
      if (token->start_byte == start_byte) {
        return i;
      }
    }
    return IREE_HOST_SIZE_MAX;
  }

  // Large windows: binary search for O(log W) complexity.
  iree_host_size_t lo = 0;
  iree_host_size_t hi = count;
  while (lo < hi) {
    iree_host_size_t mid = lo + (hi - lo) / 2;
    iree_tokenizer_bpe_window_token_t* token =
        iree_tokenizer_bpe_window_at(state, model, mid);
    if (token->start_byte < start_byte) {
      lo = mid + 1;
    } else if (token->start_byte > start_byte) {
      hi = mid;
    } else {
      return mid;  // Found.
    }
  }
  return IREE_HOST_SIZE_MAX;  // Not found (stale heap entry).
}

//===----------------------------------------------------------------------===//
// Merge Operations
//===----------------------------------------------------------------------===//

// Adds a merge candidate to the heap if a merge exists for the pair.
// The heap entry stores the left token's start_byte (stable across window
// compaction) rather than its window index (which shifts when tokens merge).
void iree_tokenizer_bpe_maybe_add_merge(iree_tokenizer_bpe_state_t* state,
                                        const iree_tokenizer_bpe_model_t* model,
                                        iree_host_size_t position) {
  if (position + 1 >= state->window.count) return;

  iree_tokenizer_bpe_window_token_t* left =
      iree_tokenizer_bpe_window_at(state, model, position);
  iree_tokenizer_bpe_window_token_t* right =
      iree_tokenizer_bpe_window_at(state, model, position + 1);

  if (left->token_id < 0 || right->token_id < 0) return;

  iree_tokenizer_merge_hash_result_t merge =
      iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, left->token_id,
                                             right->token_id);
  if (iree_tokenizer_merge_hash_result_is_valid(merge)) {
    // Store left token's start_byte - this is stable even as window compacts.
    iree_tokenizer_bpe_heap_entry_t entry = {merge.rank, left->start_byte};
    iree_tokenizer_bpe_heap_push(&state->heap, entry);
  }
}

// Applies all valid merges from the heap until no valid merges remain.
// Merges are applied in rank order (lowest rank = highest priority).
// Uses lazy invalidation: stale heap entries (where the left token no longer
// exists at that byte position, or its right neighbor changed) are simply
// discarded on pop.
void iree_tokenizer_bpe_apply_pending_merges(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  while (!iree_tokenizer_bpe_heap_is_empty(&state->heap)) {
    iree_tokenizer_bpe_heap_entry_t entry =
        iree_tokenizer_bpe_heap_peek(&state->heap);

    // Find the left token by its start_byte. This is O(window_count) but
    // window is small (bounded by 2*max_token_length, typically 256-512).
    iree_host_size_t position = iree_tokenizer_bpe_window_find_by_start_byte(
        state, model, entry.left_start_byte);

    // Stale entry: no token starts at this byte position anymore.
    if (position == IREE_HOST_SIZE_MAX) {
      iree_tokenizer_bpe_heap_pop(&state->heap);
      continue;
    }

    // Stale entry: no right neighbor (token is at end of window).
    if (position + 1 >= state->window.count) {
      iree_tokenizer_bpe_heap_pop(&state->heap);
      continue;
    }

    iree_tokenizer_bpe_window_token_t* left =
        iree_tokenizer_bpe_window_at(state, model, position);
    iree_tokenizer_bpe_window_token_t* right =
        iree_tokenizer_bpe_window_at(state, model, position + 1);

    // Look up the current merge for these tokens.
    iree_tokenizer_merge_hash_result_t merge =
        iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash,
                                               left->token_id, right->token_id);

    // Stale entry: tokens changed or merge doesn't match recorded rank.
    if (!iree_tokenizer_merge_hash_result_is_valid(merge) ||
        merge.rank != entry.rank) {
      iree_tokenizer_bpe_heap_pop(&state->heap);
      continue;
    }

    // Apply the merge: combine left+right into left, remove right.
    iree_tokenizer_bpe_heap_pop(&state->heap);
    left->token_id = merge.result_id;
    left->end_byte = right->end_byte;
    iree_tokenizer_bpe_window_remove(state, model, position + 1);

    // After merging, only two adjacencies could have new merge candidates:
    // - (position-1, position): left neighbor with merged token
    // - (position, position+1): merged token with its new right neighbor
    // Note: position+1 is now the old position+2 after removal.
    if (position > 0) {
      iree_tokenizer_bpe_maybe_add_merge(state, model, position - 1);
    }
    if (position < state->window.count - 1) {
      iree_tokenizer_bpe_maybe_add_merge(state, model, position);
    }
  }
}

// Emits all frozen tokens from the window front.
// A token ending at position p is frozen when current_byte >= p +
// max_token_length. Before emitting, applies all pending merges to ensure
// correctness. Returns false if output fills before all frozen tokens are
// emitted.
bool iree_tokenizer_bpe_emit_frozen_tokens(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model,
    iree_host_size_t current_byte_position,
    iree_tokenizer_bpe_output_cursor_t* cursor) {
  iree_host_size_t max_token_length = model->max_token_length;

  while (state->window.count > 0) {
    iree_tokenizer_bpe_window_token_t* front =
        iree_tokenizer_bpe_window_at(state, model, 0);

    // Check if the front token is frozen (can't be affected by future input).
    if (front->end_byte + max_token_length > current_byte_position + 1) {
      break;  // Not frozen yet.
    }

    // Apply all merges before emitting - the front token may change.
    iree_tokenizer_bpe_apply_pending_merges(state, model);

    // Re-fetch front (may have changed due to merges).
    front = iree_tokenizer_bpe_window_at(state, model, 0);

    // Re-check frozen condition after merges.
    if (front->end_byte + max_token_length > current_byte_position + 1) {
      break;  // Merge extended the token, no longer frozen.
    }

    // Emit the frozen token.
    if (!iree_tokenizer_bpe_emit_and_track(state, cursor, front->token_id,
                                           front->start_byte,
                                           front->end_byte)) {
      return false;  // Output full.
    }
    iree_tokenizer_bpe_window_pop_front(state, model);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Trie Helpers
//===----------------------------------------------------------------------===//

// Finds the longest token in the trie starting at |text|.
// Returns the token ID and length, or -1 and 0 if no match.
static void iree_tokenizer_bpe_trie_longest_match(
    const iree_tokenizer_bpe_model_t* model, iree_string_view_t text,
    int32_t* out_token_id, iree_host_size_t* out_length) {
  *out_token_id = -1;
  *out_length = 0;

  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  for (iree_host_size_t i = 0; i < text.size; ++i) {
    if (!iree_tokenizer_trie_cursor_advance(&cursor, (uint8_t)text.data[i])) {
      break;  // No more matches possible.
    }
    int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (token_id >= 0) {
      // Found a token ending here - remember it (greedy longest match).
      *out_token_id = token_id;
      *out_length = i + 1;
    }
  }
}
