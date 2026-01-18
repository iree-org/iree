// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/unigram.h"

#include <float.h>
#include <math.h>

#include "iree/tokenizer/vocab_internal.h"
#include "iree/tokenizer/vocab_trie.h"

//===----------------------------------------------------------------------===//
// Unigram Encoder State
//===----------------------------------------------------------------------===//

// Maximum input word length in bytes for Viterbi DP.
// Used for pre-allocated scratch buffers to avoid stack overflow on
// embedded/WASM platforms where stack size may be limited (8-64KB).
#define IREE_TOKENIZER_UNIGRAM_MAX_WORD_LENGTH 1024

// Viterbi cell for dynamic programming.
// Defined here (before state struct) so state can include the scratch buffer.
typedef struct iree_tokenizer_unigram_cell_t {
  float best_score;         // Best cumulative log probability to reach here.
  int32_t best_id;          // Token ID of the best subword ending here.
  int32_t prev_position;    // Previous position in optimal path (-1 = start).
  iree_host_size_t length;  // Length of best subword (in bytes).
} iree_tokenizer_unigram_cell_t;

struct iree_tokenizer_unigram_state_t {
  iree_allocator_t allocator;
  const iree_tokenizer_vocab_t* vocab;

  // Log probability scores indexed by token ID.
  // scores[token_id] = log(P(token_id)).
  float* scores;
  iree_host_size_t score_count;

  // Score for unknown token (used as penalty for byte fallback).
  float unk_score;

  // Prefix trie for O(N²) Viterbi traversal.
  // Owned by this state, built from vocab during allocation.
  iree_tokenizer_vocab_trie_t* trie;

  // Scratch buffers for Viterbi DP (heap-allocated to avoid stack overflow).
  // Pre-allocated once during state creation for MAX_WORD_LENGTH.
  iree_tokenizer_unigram_cell_t* dp_buffer;  // Size: MAX_WORD_LENGTH + 1.
  int32_t* backtrack_buffer;                 // Size: MAX_WORD_LENGTH.
};

iree_status_t iree_tokenizer_unigram_state_allocate(
    const iree_tokenizer_vocab_t* vocab, const float* scores,
    iree_host_size_t score_count, float unk_score, iree_allocator_t allocator,
    iree_tokenizer_unigram_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(scores);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = NULL;

  iree_host_size_t vocab_count = iree_tokenizer_vocab_capacity(vocab);
  if (score_count != vocab_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "score array size (%" PRIhsz
                            ") does not match vocab capacity (%" PRIhsz ")",
                            score_count, vocab_count);
  }

  // Allocate state (zeroed so state_free handles partial initialization).
  iree_tokenizer_unigram_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->allocator = allocator;
  state->vocab = vocab;
  state->unk_score = unk_score;
  state->score_count = vocab_count;

  // Copy scores array.
  iree_status_t status = iree_allocator_malloc(
      allocator, vocab_count * sizeof(float), (void**)&state->scores);
  if (iree_status_is_ok(status)) {
    memcpy(state->scores, scores, vocab_count * sizeof(float));
  }

  // Build prefix trie for O(N²) Viterbi traversal.
  if (iree_status_is_ok(status)) {
    iree_host_size_t array_size =
        vocab->max_token_id >= 0 ? (iree_host_size_t)(vocab->max_token_id + 1)
                                 : 0;
    iree_const_byte_span_t string_span = {vocab->string_data,
                                          vocab->string_size};
    status = iree_tokenizer_vocab_trie_build(
        vocab->tokens, array_size, string_span, allocator, &state->trie);
  }

  // Allocate Viterbi DP scratch buffer (heap instead of stack to avoid overflow
  // on embedded/WASM platforms with limited stack size).
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(allocator,
                              (IREE_TOKENIZER_UNIGRAM_MAX_WORD_LENGTH + 1) *
                                  sizeof(*state->dp_buffer),
                              (void**)&state->dp_buffer);
  }

  // Allocate backtrack scratch buffer.
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator,
                                   IREE_TOKENIZER_UNIGRAM_MAX_WORD_LENGTH *
                                       sizeof(*state->backtrack_buffer),
                                   (void**)&state->backtrack_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_state = state;
  } else {
    iree_tokenizer_unigram_state_free(state);
  }
  return status;
}

void iree_tokenizer_unigram_state_free(iree_tokenizer_unigram_state_t* state) {
  if (!state) return;
  iree_allocator_t allocator = state->allocator;

  if (state->backtrack_buffer) {
    iree_allocator_free(allocator, state->backtrack_buffer);
  }
  if (state->dp_buffer) {
    iree_allocator_free(allocator, state->dp_buffer);
  }
  if (state->trie) {
    iree_tokenizer_vocab_trie_free(state->trie);
  }
  if (state->scores) {
    iree_allocator_free(allocator, state->scores);
  }
  iree_allocator_free(allocator, state);
}

//===----------------------------------------------------------------------===//
// Viterbi Algorithm Implementation
//===----------------------------------------------------------------------===//

// Gets score for a token ID with bounds checking.
static float iree_tokenizer_unigram_get_score(
    const iree_tokenizer_unigram_state_t* state, int32_t token_id) {
  if (token_id < 0 || (iree_host_size_t)token_id >= state->score_count) {
    return state->unk_score;
  }
  return state->scores[token_id];
}

iree_status_t iree_tokenizer_unigram_encode_word(
    const iree_tokenizer_unigram_state_t* state, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_ids);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;

  if (word.size == 0) {
    return iree_ok_status();
  }

  if (word.size > IREE_TOKENIZER_UNIGRAM_MAX_WORD_LENGTH) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "word exceeds maximum length for Unigram encoding "
                            "(%" PRIhsz " > %d bytes)",
                            word.size, IREE_TOKENIZER_UNIGRAM_MAX_WORD_LENGTH);
  }

  const iree_tokenizer_vocab_t* vocab = state->vocab;

  // Use pre-allocated DP buffer from state (avoids 20KB+ stack allocation).
  // dp[i] = best way to tokenize word[0:i].
  iree_tokenizer_unigram_cell_t* dp = state->dp_buffer;

  // Initialize: position 0 has no tokens, score 0.
  dp[0].best_score = 0.0f;
  dp[0].best_id = -1;
  dp[0].prev_position = -1;
  dp[0].length = 0;

  // Initialize rest as unreachable.
  for (iree_host_size_t i = 1; i <= word.size; ++i) {
    dp[i].best_score = -FLT_MAX;
    dp[i].best_id = -1;
    dp[i].prev_position = -1;
    dp[i].length = 0;
  }

  // Forward pass: for each position, use trie to find all matching tokens.
  // This is O(N²) instead of the O(N³) hash-based approach because a single
  // trie traversal finds ALL matching prefixes in O(L) time.
  const iree_tokenizer_vocab_trie_t* trie = state->trie;
  for (iree_host_size_t start = 0; start < word.size; ++start) {
    // Skip unreachable positions (can't continue from here).
    if (dp[start].best_score == -FLT_MAX) {
      continue;
    }

    // Traverse trie from this position, finding all matching tokens.
    iree_tokenizer_trie_cursor_t cursor;
    iree_tokenizer_trie_cursor_reset(&cursor, trie);

    for (iree_host_size_t i = start; i < word.size; ++i) {
      // Try to advance by the next byte.
      if (!iree_tokenizer_trie_cursor_advance(&cursor, (uint8_t)word.data[i])) {
        break;  // No more matches possible from this position.
      }

      // Check if current position is a token boundary.
      int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
      if (token_id >= 0) {
        iree_host_size_t end = i + 1;
        iree_host_size_t length = end - start;
        float token_score = iree_tokenizer_unigram_get_score(state, token_id);
        float new_score = dp[start].best_score + token_score;

        if (new_score > dp[end].best_score) {
          dp[end].best_score = new_score;
          dp[end].best_id = token_id;
          dp[end].prev_position = (int32_t)start;
          dp[end].length = length;
        }
      }
    }
  }

  // Check if we reached the end.
  if (dp[word.size].best_id < 0) {
    // No valid tokenization found. Fall back to byte-level encoding or UNK.
    iree_tokenizer_special_ids_t special_ids =
        iree_tokenizer_vocab_special_ids(vocab);

    // Try byte-level fallback.
    iree_host_size_t byte_count = 0;
    for (iree_host_size_t i = 0; i < word.size; ++i) {
      char byte_str[8];
      snprintf(byte_str, sizeof(byte_str), "<0x%02X>",
               (unsigned char)word.data[i]);
      int32_t byte_id =
          iree_tokenizer_vocab_lookup(vocab, iree_make_cstring_view(byte_str));

      if (byte_id >= 0) {
        if (byte_count >= max_ids) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "output buffer too small");
        }
        out_ids[byte_count++] = byte_id;
      } else if (special_ids.unk >= 0) {
        // No byte token, use UNK for whole word.
        if (max_ids < 1) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "output buffer too small");
        }
        out_ids[0] = special_ids.unk;
        *out_count = 1;
        return iree_ok_status();
      } else {
        return iree_make_status(IREE_STATUS_NOT_FOUND,
                                "cannot encode word: no matching tokens, no "
                                "byte fallback, and no UNK token");
      }
    }
    *out_count = byte_count;
    return iree_ok_status();
  }

  // Backtrack to collect tokens in reverse order.
  // Use pre-allocated buffer from state (avoids 4KB stack allocation).
  int32_t* backtrack_ids = state->backtrack_buffer;
  iree_host_size_t backtrack_count = 0;

  int32_t position = (int32_t)word.size;
  while (position > 0) {
    if (backtrack_count >= IREE_TOKENIZER_UNIGRAM_MAX_WORD_LENGTH) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many tokens in Unigram output");
    }
    backtrack_ids[backtrack_count++] = dp[position].best_id;
    position = dp[position].prev_position;
  }

  // Reverse into output array.
  if (backtrack_count > max_ids) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "output buffer too small for Unigram tokens");
  }

  for (iree_host_size_t i = 0; i < backtrack_count; ++i) {
    out_ids[i] = backtrack_ids[backtrack_count - 1 - i];
  }
  *out_count = backtrack_count;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Unigram Tokenizer (unified interface)
//===----------------------------------------------------------------------===//

// Extended tokenizer structure for Unigram.
typedef struct iree_tokenizer_unigram_t {
  iree_tokenizer_t base;
  iree_tokenizer_unigram_state_t* state;  // Owned.
  float* scores_storage;                  // Owned score array.
} iree_tokenizer_unigram_t;

static void iree_tokenizer_unigram_destroy_impl(iree_tokenizer_t* tokenizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_unigram_t* unigram = (iree_tokenizer_unigram_t*)tokenizer;
  // Destroy Unigram-specific state.
  if (unigram->state) {
    iree_tokenizer_unigram_state_free(unigram->state);
  }
  // Note: scores_storage is freed by state_free (it copies the array).
  if (unigram->scores_storage) {
    iree_allocator_free(unigram->base.allocator, unigram->scores_storage);
  }
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_unigram_encode_word_impl(
    const iree_tokenizer_t* tokenizer, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count) {
  const iree_tokenizer_unigram_t* unigram =
      (const iree_tokenizer_unigram_t*)tokenizer;
  return iree_tokenizer_unigram_encode_word(unigram->state, word, out_ids,
                                            max_ids, out_count);
}

static const iree_tokenizer_vtable_t iree_tokenizer_unigram_vtable = {
    .destroy = iree_tokenizer_unigram_destroy_impl,
    .encode_word = iree_tokenizer_unigram_encode_word_impl,
};

iree_status_t iree_tokenizer_unigram_allocate(
    iree_tokenizer_vocab_t* vocab, float* scores, iree_host_size_t score_count,
    float unk_score, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(scores);
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the extended tokenizer struct.
  iree_tokenizer_unigram_t* tokenizer = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator, sizeof(iree_tokenizer_unigram_t), (void**)&tokenizer);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_free(vocab);
    iree_allocator_free(allocator, scores);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Initialize Unigram-specific fields to NULL before any fallible calls.
  // This ensures free_impl's NULL checks work correctly on error paths.
  tokenizer->state = NULL;
  tokenizer->scores_storage = NULL;

  // Initialize base (stores vocab - from here, iree_tokenizer_free handles it).
  iree_tokenizer_initialize(&tokenizer->base, &iree_tokenizer_unigram_vtable,
                            allocator, vocab, /*transform=*/NULL,
                            /*decoder=*/NULL, /*postprocessor=*/NULL);

  // Store scores for later cleanup.
  tokenizer->scores_storage = scores;

  // Create Unigram state.
  status = iree_tokenizer_unigram_state_allocate(
      vocab, scores, score_count, unk_score, allocator, &tokenizer->state);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_free(&tokenizer->base);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_tokenizer = &tokenizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
