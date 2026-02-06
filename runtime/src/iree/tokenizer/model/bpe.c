// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE (Byte-Pair Encoding) tokenization using sliding window algorithm.
//
// Standard BPE applies merge rules in priority order (lowest rank first), not
// greedy longest-match. The naive algorithm is O(n*M) where M is merge count.
// We use a sliding window with min-heap to achieve O(n log L) where L is the
// maximum token length.
//
// CORRECTNESS PROOF (Frozen Token Theorem):
// A token whose byte range ends at position p cannot be affected by any bytes
// at position p + L or beyond, where L is the maximum token length. This is
// because:
// - A merge can only combine adjacent tokens
// - The longest possible token spans L bytes
// - For a token ending at p to change, a merge would need bytes beyond p
// - But any token starting after p extends at most L bytes, reaching p + L
// Therefore, once we've processed L bytes past p, the token at p is "frozen"
// and can be safely emitted.
//
// ALGORITHM:
// - Maintain a sliding window of at most 2*L tokens
// - Use a min-heap of merge candidates ordered by rank (lowest = highest
// priority)
// - For each input byte: create token, add merge candidate with left neighbor
// - Apply all possible merges in rank order (heap pop, validate, merge)
// - Emit frozen tokens (end_byte <= current_position - L + 1)
// - At segment end, flush remaining tokens
//
// COMPLEXITY:
// - Time: O(n log L) where n = input length, L = max token length
// - Memory: O(L) bounded, independent of input size

#include "iree/tokenizer/model/bpe.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/tokenizer/byte_level_tables.h"
#include "iree/tokenizer/model/bpe_heap.h"
#include "iree/tokenizer/model/bpe_internal.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

static const iree_tokenizer_model_vtable_t iree_tokenizer_bpe_model_vtable;
static void iree_tokenizer_bpe_model_destroy(
    iree_tokenizer_model_t* base_model);

//===----------------------------------------------------------------------===//
// BPE Model Allocation
//===----------------------------------------------------------------------===//

// Encodes a Unicode codepoint (0-0x143) as UTF-8.
// Returns the number of bytes written (1-2 for ByteLevel codepoints).
// Buffer must have at least 2 bytes of space.
static inline iree_host_size_t iree_tokenizer_bpe_encode_utf8(
    uint16_t codepoint, char* buffer) {
  if (IREE_LIKELY(codepoint <= 0x7F)) {
    buffer[0] = (char)codepoint;
    return 1;
  } else {
    // Codepoints 0x80-0x143 encode as 2-byte UTF-8.
    buffer[0] = (char)(0xC0 | (codepoint >> 6));
    buffer[1] = (char)(0x80 | (codepoint & 0x3F));
    return 2;
  }
}

// Builds the byte_to_token and byte_fallback_token lookup tables.
// byte_to_token: maps each byte value to its single-byte vocabulary token.
// byte_fallback_token: maps each byte value to its "<0xNN>" fallback token.
static void iree_tokenizer_bpe_build_byte_to_token(
    iree_tokenizer_bpe_model_t* model) {
  const bool byte_level =
      iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);

  for (int byte_value = 0; byte_value < 256; ++byte_value) {
    char utf8_buffer[4];
    iree_host_size_t utf8_length;

    if (byte_level) {
      // ByteLevel: map byte through GPT-2 table, encode as UTF-8.
      uint16_t codepoint = iree_tokenizer_byte_level_mapping[byte_value];
      utf8_length = iree_tokenizer_bpe_encode_utf8(codepoint, utf8_buffer);
    } else {
      // Direct: byte is used as-is.
      utf8_buffer[0] = (char)byte_value;
      utf8_length = 1;
    }

    // Look up in vocabulary.
    model->byte_to_token[byte_value] = iree_tokenizer_vocab_lookup(
        model->vocab, iree_make_string_view(utf8_buffer, utf8_length));

    // Precompute byte fallback token: "<0xNN>".
    char fallback_token[8];
    snprintf(fallback_token, sizeof(fallback_token), "<0x%02X>", byte_value);
    model->byte_fallback_token[byte_value] = iree_tokenizer_vocab_lookup(
        model->vocab, iree_make_cstring_view(fallback_token));
  }
}

iree_status_t iree_tokenizer_bpe_model_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_bpe_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_model);
  *out_model = NULL;

  iree_tokenizer_bpe_model_t* model = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*model), (void**)&model));
  memset(model, 0, sizeof(*model));

  model->allocator = allocator;
  model->vocab = vocab;
  model->flags = flags;

  // Get max token length for window sizing.
  model->max_token_length = iree_tokenizer_vocab_max_token_length(vocab);
  if (model->max_token_length == 0) {
    model->max_token_length = 1;  // Safety minimum.
  }

  iree_status_t status = iree_ok_status();

  // Window capacity is 2 * max_token_length (frozen theorem), rounded up to
  // power of 2 for fast modulo via bitmask.
  iree_host_size_t min_window_capacity = 0;
  if (!iree_host_size_checked_mul(2, model->max_token_length,
                                  &min_window_capacity)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "max_token_length overflow in window capacity");
  }
  if (iree_status_is_ok(status)) {
    model->window_capacity =
        iree_math_round_up_to_pow2_u64((uint64_t)min_window_capacity);
    model->window_capacity_mask = model->window_capacity - 1;
  }

  // Heap capacity bounds proof:
  //
  // Let L = max_token_length, H = heap entries at apply_pending_merges call,
  // W = window tokens at that call.
  //
  // Lemma 1: Peak heap during apply_pending_merges = H + (W - 1).
  //   Each merge pops 1, adds ≤2 (net +1). Max merges = W - 1.
  //
  // Lemma 2: H ≤ L - 1 (except first call where H = L).
  //   Heap drains after each apply_pending_merges. Between drains, add 1
  //   entry per byte. Freeze when front.end_byte + L ≤ current + 1.
  //   Max bytes between freezes = L - 1 (front.end_byte can grow by L - 1).
  //
  // Lemma 3: W ≤ 2L - 1.
  //   For front NOT frozen: front.end_byte > current + 1 - L.
  //   Since front.end_byte ≤ window_start + L, span < 2L, so W < 2L.
  //
  // First call: H = L, W = L + 1. Peak = L + L = 2L.
  // Subsequent: H ≤ L - 1, W ≤ 2L - 1. Peak ≤ (L-1) + (2L-2) = 3L - 3.
  //
  // Maximum is 3L - 3 < 3L, so capacity = 3L suffices.
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(3, model->max_token_length,
                                  &model->heap_capacity)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "max_token_length overflow in heap capacity");
  }

  if (iree_status_is_ok(status)) {
    // Vocab capacity for backtracking table sizing.
    model->vocab_capacity = iree_tokenizer_vocab_capacity(vocab);

    // Backtracking path capacities.
    // Segments up to this size use the O(n) backtracking algorithm.
    // Larger segments fall back to the O(n log L) window+heap path.
    // Capped at 4095 so the bitfield (ceil_div(4096, 64) = 64 words) fits in
    // a single uint64_t dirty mask for O(1) amortized per-segment init.
    iree_host_size_t min_backtrack = 0;
    if (!iree_host_size_checked_mul(16, model->max_token_length,
                                    &min_backtrack)) {
      // Overflow implies very large token length; clamp to maximum.
      min_backtrack = 4095;
    }
    if (min_backtrack < 2048) min_backtrack = 2048;
    if (min_backtrack > 4095) min_backtrack = 4095;
    model->max_backtrack_segment_bytes = min_backtrack;
    // Stack holds at most one token per byte (worst case: all single-byte
    // tokens).
    model->backtrack_stack_capacity = model->max_backtrack_segment_bytes;
    // Bitfield tracks one bit per byte position, including the end position
    // (positions 0..segment_size inclusive, so segment_size+1 bits needed).
    model->backtrack_bitfield_capacity =
        iree_host_size_ceil_div(model->max_backtrack_segment_bytes + 1, 64);

    // Word cache: only allocate when the caller has signaled that the
    // pre-tokenizer produces word-level segments (ENABLE_WORD_CACHE flag).
    // Without word-level splitting, input flows through the streaming partial
    // path and the cache is never consulted — allocating it wastes ~40KB of
    // state with zero benefit. Additionally, the cache exploits Zipf's law
    // which only applies with sufficient vocabulary diversity (>= 256 tokens).
    if (iree_any_bit_set(flags, IREE_TOKENIZER_BPE_FLAG_ENABLE_WORD_CACHE) &&
        model->vocab_capacity >= IREE_TOKENIZER_BPE_CACHE_MIN_VOCAB_SIZE) {
      model->cache_capacity = IREE_TOKENIZER_BPE_CACHE_CAPACITY;
    } else {
      model->cache_capacity = 0;
    }
    model->cache_capacity_mask =
        model->cache_capacity > 0 ? model->cache_capacity - 1 : 0;
  }

  // Compute state size and offsets with overflow-checked math.
  // Memory layout:
  //   [state struct]
  //   [window tokens]       - window+heap path
  //   [heap entries]        - window+heap path
  //   [backtrack stack]     - backtracking path
  //   [backtrack bitfield]  - backtracking path
  //   [word cache]          - repeated-word acceleration (0 if disabled)
  iree_host_size_t state_size = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_tokenizer_bpe_state_t), &state_size,
        IREE_STRUCT_FIELD(model->window_capacity,
                          iree_tokenizer_bpe_window_token_t,
                          &model->window_offset),
        IREE_STRUCT_FIELD(model->heap_capacity, iree_tokenizer_bpe_heap_entry_t,
                          &model->heap_offset),
        IREE_STRUCT_FIELD(model->backtrack_stack_capacity,
                          iree_tokenizer_bpe_backtrack_entry_t,
                          &model->backtrack_stack_offset),
        IREE_STRUCT_FIELD(model->backtrack_bitfield_capacity, uint64_t,
                          &model->backtrack_bitfield_offset),
        IREE_STRUCT_FIELD(model->cache_capacity,
                          iree_tokenizer_bpe_cache_entry_t,
                          &model->cache_offset));
  }

  if (iree_status_is_ok(status)) {
    iree_tokenizer_model_initialize(&model->base,
                                    &iree_tokenizer_bpe_model_vtable,
                                    state_size, IREE_SV("BPE"));

    // Build byte-to-token lookup table.
    iree_tokenizer_bpe_build_byte_to_token(model);

    // Build prefix trie for token lookup.
    const iree_tokenizer_token_t* tokens = iree_tokenizer_vocab_tokens(vocab);
    iree_host_size_t token_count = iree_tokenizer_vocab_capacity(vocab);
    iree_const_byte_span_t string_table =
        iree_tokenizer_vocab_string_table(vocab);
    status = iree_tokenizer_vocab_trie_build(tokens, token_count, string_table,
                                             allocator, &model->trie);
  }

  // Build merge hash table.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_merge_hash_build(vocab, allocator,
                                                   &model->merge_hash);
  }

  // Build backtracking tables (split_table, next_prefix_match, effective_rank).
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_bpe_build_backtrack_tables(model);
  }

  if (iree_status_is_ok(status)) {
    *out_model = &model->base;
  } else {
    iree_tokenizer_bpe_model_destroy(&model->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_bpe_model_set_end_of_word_suffix(
    iree_tokenizer_model_t* base_model, iree_string_view_t suffix) {
  IREE_ASSERT_ARGUMENT(base_model);
  iree_tokenizer_bpe_model_t* model = (iree_tokenizer_bpe_model_t*)base_model;
  if (suffix.size > sizeof(model->end_of_word_suffix)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "end_of_word_suffix too long (%" PRIhsz
                            " bytes, max %zu)",
                            suffix.size, sizeof(model->end_of_word_suffix));
  }
  memcpy(model->end_of_word_suffix, suffix.data, suffix.size);
  model->end_of_word_suffix_length = suffix.size;
  return iree_ok_status();
}

static void iree_tokenizer_bpe_model_destroy(
    iree_tokenizer_model_t* base_model) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_bpe_model_t* model = (iree_tokenizer_bpe_model_t*)base_model;
  iree_allocator_t allocator = model->allocator;

  iree_tokenizer_vocab_trie_free(model->trie);
  iree_tokenizer_vocab_merge_hash_free(model->merge_hash);
  iree_allocator_free(allocator, model->backtrack_tables.slab);
  iree_allocator_free(allocator, model);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// BPE State Operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_bpe_state_initialize(
    const iree_tokenizer_model_t* base_model, void* storage,
    iree_tokenizer_model_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)base_model;
  iree_tokenizer_bpe_state_t* state = (iree_tokenizer_bpe_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.model = base_model;
  state->phase = IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;
  state->window.count = 0;
  state->window.start = 0;
  state->segment.byte_position = 0;
  state->fast_path_pending_token_id = -1;
  state->last_emitted_token_id = -1;

  // Mark all bitfield words as dirty so the first backtrack_encode call
  // initializes them to UINT64_MAX (the bitfield storage is uninitialized).
  state->backtrack.dirty_mask =
      (model->backtrack_bitfield_capacity >= 64)
          ? UINT64_MAX
          : ((uint64_t)1 << model->backtrack_bitfield_capacity) - 1;

  // Initialize heap with trailing buffer storage.
  iree_tokenizer_bpe_heap_entry_t* heap_storage =
      iree_tokenizer_bpe_state_heap(state, model);
  iree_tokenizer_bpe_heap_initialize(&state->heap, heap_storage,
                                     model->heap_capacity);

  // Zero the word cache so key_hash == 0 correctly marks empty slots.
  if (model->cache_capacity > 0) {
    memset(iree_tokenizer_bpe_state_cache(state, model), 0,
           model->cache_capacity * sizeof(iree_tokenizer_bpe_cache_entry_t));
  }

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_bpe_state_deinitialize(
    iree_tokenizer_model_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// BPE Encoding (Sliding Window with Min-Heap)
//===----------------------------------------------------------------------===//

// Handles a byte that could not be encoded via the trie.
int32_t iree_tokenizer_bpe_handle_unknown_byte(
    const iree_tokenizer_bpe_model_t* model, uint8_t byte,
    int32_t last_emitted_token_id) {
  const iree_tokenizer_vocab_t* vocab = model->vocab;
  iree_tokenizer_special_ids_t special_ids =
      iree_tokenizer_vocab_special_ids(vocab);

  if (iree_all_bits_set(model->flags,
                        IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK)) {
    if (special_ids.unk < 0) return -1;
    if (iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_FUSE_UNK) &&
        last_emitted_token_id == special_ids.unk) {
      return -1;
    }
    return special_ids.unk;
  }

  // Try byte fallback token: <0xNN> (precomputed during model construction).
  int32_t token_id = model->byte_fallback_token[byte];
  if (token_id >= 0) return token_id;

  if (special_ids.unk < 0) return -1;
  if (iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_FUSE_UNK) &&
      last_emitted_token_id == special_ids.unk) {
    return -1;
  }
  return special_ids.unk;
}

//===----------------------------------------------------------------------===//
// BPE Token String Lookup (for decode)
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_bpe_get_token_string(
    const iree_tokenizer_model_t* base_model,
    iree_tokenizer_token_id_t token_id, iree_string_view_t* out_string) {
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)base_model;

  iree_string_view_t text =
      iree_tokenizer_vocab_token_text(model->vocab, token_id);
  if (text.size == 0 && token_id >= 0) {
    iree_host_size_t capacity = iree_tokenizer_vocab_capacity(model->vocab);
    if ((iree_host_size_t)token_id >= capacity) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "token ID %d out of range (capacity %" PRIhsz ")",
                              token_id, capacity);
    }
  }

  *out_string = text;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BPE Model VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_model_vtable_t iree_tokenizer_bpe_model_vtable = {
    .destroy = iree_tokenizer_bpe_model_destroy,
    .state_initialize = iree_tokenizer_bpe_state_initialize,
    .state_deinitialize = iree_tokenizer_bpe_state_deinitialize,
    .state_encode = iree_tokenizer_bpe_state_encode,
    .state_finalize = iree_tokenizer_bpe_state_finalize,
    .state_has_pending = iree_tokenizer_bpe_state_has_pending,
    .state_reclaim = iree_tokenizer_bpe_state_reclaim,
    .get_token_string = iree_tokenizer_bpe_get_token_string,
};
