// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unigram tokenization using incremental Viterbi dynamic programming.
//
// ALGORITHM:
// For each segment, process in chunk_size-byte chunks:
//   1. Initialize DP: best_score[0] = 0, all others = -FLT_MAX
//   2. For each byte position (advancing by UTF-8 character):
//      a. Skip unreachable positions (best_score == -FLT_MAX)
//      b. Walk prefix trie from position, finding all matching tokens
//         (capped at chunk boundary)
//      c. For each match: update DP if score improvement found
//      d. If no single-char token covers this position, insert UNK candidate
//   3. If chunk end unreachable: byte fallback or single UNK
//   4. Backtrack from end: collect tokens in reverse, expand UNK to bytes,
//      fuse consecutive UNKs
//   5. Emit tokens from pending buffer
//
// STREAMING:
// Segments are processed incrementally in chunk_size-byte chunks.
// Each chunk runs exact Viterbi. Partial segments (from pipeline ring
// overflow) are processed the same way, with state_reclaim returning
// committed bytes so the pipeline can free ring space. DP buffers are
// sized to chunk_size (at least max_token_length), giving O(L) state.
//
// COMPLEXITY:
// - Time: O(n * L) per segment, where n = byte length, L = max token length
// - Memory: O(L) for DP tables and pending buffer

#include "iree/tokenizer/model/unigram.h"

#include <float.h>
#include <stdio.h>
#include <string.h>

#include "iree/tokenizer/vocab/vocab_trie.h"

// Minimum chunk size for Viterbi processing (in bytes). Ensures exact Viterbi
// for segments shorter than this, even when the vocabulary has short tokens.
// With 128 bytes: any word/segment up to 128 bytes is processed in one pass
// without forced chunk boundaries. DP state is ~4KB.
#define IREE_TOKENIZER_UNIGRAM_MIN_CHUNK_SIZE 128

//===----------------------------------------------------------------------===//
// Pending Token Entry
//===----------------------------------------------------------------------===//

// Pre-computed sub-token for deferred emission.
// Byte offsets are relative to the segment start (not the transform buffer).
// The segment_base_offset is added when emitting to produce absolute offsets.
typedef struct iree_tokenizer_unigram_pending_token_t {
  iree_tokenizer_token_id_t token_id;
  // Byte offsets relative to the current segment start.
  iree_host_size_t start_byte;
  iree_host_size_t end_byte;
} iree_tokenizer_unigram_pending_token_t;

//===----------------------------------------------------------------------===//
// Unigram Model Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_unigram_model_t {
  iree_tokenizer_model_t base;
  iree_allocator_t allocator;
  const iree_tokenizer_vocab_t* vocab;  // NOT owned.
  iree_tokenizer_vocab_trie_t* trie;    // OWNED.

  iree_tokenizer_token_id_t unk_token_id;
  float unk_score;

  // Pre-computed byte fallback table: byte value → <0xXX> token ID.
  // IREE_TOKENIZER_TOKEN_ID_INVALID if no such token exists.
  int32_t byte_to_token[256];

  // Maximum token text length in the vocabulary (in bytes).
  // Used for partial segment holdback: the last max_token_length bytes of a
  // partial segment are held back because future data could change them.
  iree_host_size_t max_token_length;
  // Processing chunk size (in bytes). Viterbi runs on chunks of this size.
  // Equal to max(max_token_length, IREE_TOKENIZER_UNIGRAM_MIN_CHUNK_SIZE)
  // to ensure segments shorter than the minimum are processed in one pass.
  iree_host_size_t chunk_size;
  iree_tokenizer_unigram_flags_t flags;

  // Byte offsets into state slab for trailing buffers.
  iree_host_size_t best_score_offset;
  iree_host_size_t best_token_id_offset;
  iree_host_size_t best_length_offset;
  iree_host_size_t pending_tokens_offset;
} iree_tokenizer_unigram_model_t;

//===----------------------------------------------------------------------===//
// Unigram State Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_unigram_state_t {
  iree_tokenizer_model_state_t base;

  // Pre-computed tokens for the current chunk.
  iree_host_size_t pending_count;
  iree_host_size_t pending_emit_index;

  // Streaming state for incremental segment processing.
  // byte_position: how many bytes of the current segment have been processed
  // through the Viterbi forward pass (advances by chunk_length per chunk).
  iree_host_size_t byte_position;
  // committed_position: bytes whose tokens have ALL been emitted. Reclaimable
  // by state_reclaim. Always <= byte_position.
  iree_host_size_t committed_position;
  // Transform buffer offset of the current segment. Updated on each
  // state_encode call and used to compute absolute byte offsets for output.
  iree_host_size_t segment_base_offset;

  // Trailing buffers (accessed via model offsets):
  //   float best_score[chunk_size + 1]
  //   int32_t best_token_id[chunk_size + 1]
  //   uint16_t best_length[chunk_size + 1]
  //   iree_tokenizer_unigram_pending_token_t
  //       pending_tokens[chunk_size]
} iree_tokenizer_unigram_state_t;

//===----------------------------------------------------------------------===//
// Trailing Buffer Accessors
//===----------------------------------------------------------------------===//

static inline float* iree_tokenizer_unigram_state_best_score(
    iree_tokenizer_unigram_state_t* state,
    const iree_tokenizer_unigram_model_t* model) {
  return (float*)((uint8_t*)state + model->best_score_offset);
}

static inline int32_t* iree_tokenizer_unigram_state_best_token_id(
    iree_tokenizer_unigram_state_t* state,
    const iree_tokenizer_unigram_model_t* model) {
  return (int32_t*)((uint8_t*)state + model->best_token_id_offset);
}

static inline uint16_t* iree_tokenizer_unigram_state_best_length(
    iree_tokenizer_unigram_state_t* state,
    const iree_tokenizer_unigram_model_t* model) {
  return (uint16_t*)((uint8_t*)state + model->best_length_offset);
}

static inline iree_tokenizer_unigram_pending_token_t*
iree_tokenizer_unigram_state_pending_tokens(
    iree_tokenizer_unigram_state_t* state,
    const iree_tokenizer_unigram_model_t* model) {
  return (
      iree_tokenizer_unigram_pending_token_t*)((uint8_t*)state +
                                               model->pending_tokens_offset);
}

//===----------------------------------------------------------------------===//
// UTF-8 Helper
//===----------------------------------------------------------------------===//

// Returns the byte length of a UTF-8 sequence starting at |byte|.
// Returns 1 for invalid lead bytes (treats as single-byte character).
static inline iree_host_size_t iree_tokenizer_unigram_utf8_char_length(
    uint8_t byte) {
  if (byte < 0x80) return 1;
  if ((byte & 0xE0) == 0xC0) return 2;
  if ((byte & 0xF0) == 0xE0) return 3;
  if ((byte & 0xF8) == 0xF0) return 4;
  return 1;  // Invalid lead byte, treat as single byte.
}

// Returns true if |byte| is a UTF-8 lead byte (start of a character).
static inline bool iree_tokenizer_unigram_is_utf8_lead(uint8_t byte) {
  return byte < 0x80 || byte >= 0xC0;
}

//===----------------------------------------------------------------------===//
// Byte Fallback Table Builder
//===----------------------------------------------------------------------===//

// Pre-computes the byte_to_token[256] table by looking up <0xXX> tokens.
static void iree_tokenizer_unigram_build_byte_to_token(
    iree_tokenizer_unigram_model_t* model) {
  for (int byte_value = 0; byte_value < 256; ++byte_value) {
    char byte_string[8];
    snprintf(byte_string, sizeof(byte_string), "<0x%02X>", byte_value);
    model->byte_to_token[byte_value] = iree_tokenizer_vocab_lookup(
        model->vocab,
        iree_make_string_view(byte_string, 6));  // "<0xXX>" is always 6 chars.
  }
}

//===----------------------------------------------------------------------===//
// Fallback Token Insertion
//===----------------------------------------------------------------------===//

// Inserts fallback candidates for a character position with no vocab coverage.
// For single-byte chars: inserts one byte token candidate.
// For multi-byte chars: inserts byte token candidates for each byte.
// If byte tokens unavailable, inserts UNK for the whole character.
// Returns true if any candidates were inserted.
static bool iree_tokenizer_unigram_insert_fallback_candidates(
    const iree_tokenizer_unigram_model_t* model, const uint8_t* chunk_data,
    iree_host_size_t position, iree_host_size_t char_length,
    iree_host_size_t chunk_length, float* best_score, int32_t* best_token_id,
    uint16_t* best_length) {
  // Try byte fallback first.
  if (!iree_any_bit_set(model->flags,
                        IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK)) {
    // Check if all bytes of the character have byte tokens.
    bool all_bytes_available = true;
    for (iree_host_size_t b = 0; b < char_length; ++b) {
      if (model->byte_to_token[chunk_data[position + b]] ==
          IREE_TOKENIZER_TOKEN_ID_INVALID) {
        all_bytes_available = false;
        break;
      }
    }
    if (all_bytes_available) {
      // Insert byte token candidates for each byte of the character.
      float running_score = best_score[position];
      for (iree_host_size_t b = 0; b < char_length; ++b) {
        int32_t byte_token = model->byte_to_token[chunk_data[position + b]];
        float byte_score =
            iree_tokenizer_vocab_token_score(model->vocab, byte_token);
        iree_host_size_t byte_end = position + b + 1;
        float new_score = running_score + byte_score;
        if (new_score > best_score[byte_end]) {
          best_score[byte_end] = new_score;
          best_token_id[byte_end] = byte_token;
          best_length[byte_end] = 1;
        }
        running_score = best_score[byte_end];
      }
      return true;
    }
  }

  // Fall back to UNK for the whole character.
  if (model->unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    iree_host_size_t end = position + char_length;
    float score = best_score[position] + model->unk_score;
    if (score > best_score[end]) {
      best_score[end] = score;
      best_token_id[end] = model->unk_token_id;
      best_length[end] = (uint16_t)char_length;
    }
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Viterbi Algorithm (operates on a single chunk)
//===----------------------------------------------------------------------===//

// Runs the Viterbi forward pass and backtrack for a single chunk.
// |chunk_data| points to the first byte of the chunk.
// |chunk_length| is the byte length (at most chunk_size).
// |segment_byte_offset| is the chunk's start position within the segment
// (used to compute segment-relative offsets for pending tokens).
//
// Fills the pending buffer with the optimal tokenization.
// Returns the number of pending tokens, or 0 if the chunk should be handled
// as a byte-fallback or UNK case by the caller.
static iree_host_size_t iree_tokenizer_unigram_viterbi(
    const iree_tokenizer_unigram_model_t* model,
    iree_tokenizer_unigram_state_t* state, const uint8_t* chunk_data,
    iree_host_size_t chunk_length, iree_host_size_t segment_byte_offset) {
  float* best_score = iree_tokenizer_unigram_state_best_score(state, model);
  int32_t* best_token_id =
      iree_tokenizer_unigram_state_best_token_id(state, model);
  uint16_t* best_length =
      iree_tokenizer_unigram_state_best_length(state, model);
  iree_tokenizer_unigram_pending_token_t* pending_tokens =
      iree_tokenizer_unigram_state_pending_tokens(state, model);

  // Initialize DP table.
  best_score[0] = 0.0f;
  best_token_id[0] = IREE_TOKENIZER_TOKEN_ID_INVALID;
  best_length[0] = 0;
  for (iree_host_size_t i = 1; i <= chunk_length; ++i) {
    best_score[i] = -FLT_MAX;
    best_token_id[i] = IREE_TOKENIZER_TOKEN_ID_INVALID;
    best_length[i] = 0;
  }

  // Forward pass: advance by UTF-8 character boundaries.
  iree_host_size_t position = 0;
  while (position < chunk_length) {
    iree_host_size_t char_length =
        iree_tokenizer_unigram_utf8_char_length(chunk_data[position]);
    // Clamp char_length to not exceed chunk.
    if (position + char_length > chunk_length) {
      char_length = chunk_length - position;
    }

    // Skip unreachable positions.
    if (best_score[position] == -FLT_MAX) {
      position += char_length;
      continue;
    }

    // Walk prefix trie from this position, finding all matching tokens.
    // Cap at chunk boundary — no token may extend past chunk_length.
    bool has_single_char_coverage = false;
    iree_tokenizer_trie_cursor_t cursor;
    iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

    for (iree_host_size_t i = position; i < chunk_length; ++i) {
      if (!iree_tokenizer_trie_cursor_advance(&cursor, chunk_data[i])) {
        break;  // No more prefixes possible.
      }
      int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
      if (token_id >= 0) {
        iree_host_size_t end = i + 1;
        iree_host_size_t token_length = end - position;
        float score = best_score[position] +
                      iree_tokenizer_vocab_token_score(model->vocab, token_id);
        if (score > best_score[end]) {
          best_score[end] = score;
          best_token_id[end] = token_id;
          best_length[end] = (uint16_t)token_length;
        }
        if (token_length == char_length) {
          has_single_char_coverage = true;
        }
      }
    }

    // If no vocab token covers even a single character from this position,
    // insert fallback candidates (byte tokens or UNK) to continue the DP path.
    if (!has_single_char_coverage) {
      iree_tokenizer_unigram_insert_fallback_candidates(
          model, chunk_data, position, char_length, chunk_length, best_score,
          best_token_id, best_length);
    }

    position += char_length;
  }

  // Check if chunk end is reachable.
  if (best_token_id[chunk_length] == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    // No valid tokenization found — caller handles byte fallback or UNK.
    return 0;
  }

  // Backtrack to reconstruct optimal tokenization.
  // Collect tokens in reverse order first.
  iree_host_size_t backtrack_count = 0;
  position = chunk_length;
  while (position > 0) {
    int32_t token_id = best_token_id[position];
    uint16_t length = best_length[position];
    iree_host_size_t start = position - length;

    // Store in pending buffer (reverse order, will be flipped below).
    // Offsets are relative to segment start.
    pending_tokens[backtrack_count].token_id = token_id;
    pending_tokens[backtrack_count].start_byte = segment_byte_offset + start;
    pending_tokens[backtrack_count].end_byte = segment_byte_offset + position;
    ++backtrack_count;

    position = start;
  }

  // Reverse the pending tokens to get forward order.
  for (iree_host_size_t i = 0; i < backtrack_count / 2; ++i) {
    iree_host_size_t j = backtrack_count - 1 - i;
    iree_tokenizer_unigram_pending_token_t temp = pending_tokens[i];
    pending_tokens[i] = pending_tokens[j];
    pending_tokens[j] = temp;
  }

  // Apply UNK fusion if enabled: merge consecutive UNK tokens.
  if (!iree_any_bit_set(model->flags,
                        IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK) &&
      model->unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    iree_host_size_t write_index = 0;
    for (iree_host_size_t i = 0; i < backtrack_count; ++i) {
      if (pending_tokens[i].token_id == model->unk_token_id &&
          write_index > 0 &&
          pending_tokens[write_index - 1].token_id == model->unk_token_id) {
        // Extend previous UNK to cover this one.
        pending_tokens[write_index - 1].end_byte = pending_tokens[i].end_byte;
      } else {
        pending_tokens[write_index] = pending_tokens[i];
        ++write_index;
      }
    }
    backtrack_count = write_index;
  }

  return backtrack_count;
}

// Tokenizes a single chunk, handling byte fallback and UNK for unreachable
// cases. Returns the number of pending tokens written.
static iree_host_size_t iree_tokenizer_unigram_tokenize_chunk(
    const iree_tokenizer_unigram_model_t* model,
    iree_tokenizer_unigram_state_t* state, const uint8_t* chunk_data,
    iree_host_size_t chunk_length, iree_host_size_t segment_byte_offset) {
  iree_tokenizer_unigram_pending_token_t* pending_tokens =
      iree_tokenizer_unigram_state_pending_tokens(state, model);

  // Empty chunks produce no tokens.
  if (chunk_length == 0) return 0;

  // Run Viterbi DP.
  iree_host_size_t token_count = iree_tokenizer_unigram_viterbi(
      model, state, chunk_data, chunk_length, segment_byte_offset);
  if (token_count > 0) return token_count;

  // Viterbi found no valid path — try byte fallback.
  if (!iree_any_bit_set(model->flags,
                        IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK)) {
    // Verify all bytes have fallback tokens before committing.
    bool all_bytes_available = true;
    for (iree_host_size_t i = 0; i < chunk_length; ++i) {
      if (model->byte_to_token[chunk_data[i]] ==
          IREE_TOKENIZER_TOKEN_ID_INVALID) {
        all_bytes_available = false;
        break;
      }
    }
    if (all_bytes_available) {
      for (iree_host_size_t i = 0; i < chunk_length; ++i) {
        pending_tokens[i].token_id = model->byte_to_token[chunk_data[i]];
        pending_tokens[i].start_byte = segment_byte_offset + i;
        pending_tokens[i].end_byte = segment_byte_offset + i + 1;
      }
      return chunk_length;
    }
  }

  // Final fallback: single UNK for entire chunk.
  if (model->unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    pending_tokens[0].token_id = model->unk_token_id;
    pending_tokens[0].start_byte = segment_byte_offset;
    pending_tokens[0].end_byte = segment_byte_offset + chunk_length;
    return 1;
  }

  // No UNK and no byte fallback — chunk cannot be tokenized.
  return 0;
}

// Aligns a byte position to the nearest UTF-8 character boundary at or before
// |position|. Scans backward at most 3 bytes (max UTF-8 continuation bytes).
// |data| must have at least |position| bytes accessible.
// Returns |position| if already at a character boundary, or the adjusted
// position. Never returns less than |min_position|.
static iree_host_size_t iree_tokenizer_unigram_align_utf8_boundary(
    const uint8_t* data, iree_host_size_t position,
    iree_host_size_t min_position) {
  while (position > min_position &&
         !iree_tokenizer_unigram_is_utf8_lead(data[position])) {
    --position;
  }
  return position;
}

//===----------------------------------------------------------------------===//
// Model Allocation
//===----------------------------------------------------------------------===//

static const iree_tokenizer_model_vtable_t iree_tokenizer_unigram_model_vtable;
static void iree_tokenizer_unigram_model_destroy(
    iree_tokenizer_model_t* base_model);

iree_status_t iree_tokenizer_unigram_model_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_token_id_t unk_token_id,
    float unk_score, iree_tokenizer_unigram_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_model);
  *out_model = NULL;

  // UNK is required when byte_fallback is disabled (otherwise some segments
  // would be untokenizable).
  if (iree_any_bit_set(flags, IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK) &&
      unk_token_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "UNK token required when byte_fallback disabled"));
  }

  // Derive max_token_length from the vocabulary. This bounds the DP lookback
  // and the partial segment holdback distance.
  iree_host_size_t max_token_length =
      iree_tokenizer_vocab_max_token_length(vocab);
  if (max_token_length == 0) {
    max_token_length = 1;  // Safety minimum.
  }

  // Chunk size: the number of bytes processed per Viterbi pass. Using a
  // minimum ensures exact Viterbi for typical word-sized segments even when
  // the vocabulary has short tokens (e.g., single characters + [UNK]).
  iree_host_size_t chunk_size = max_token_length;
  if (chunk_size < IREE_TOKENIZER_UNIGRAM_MIN_CHUNK_SIZE) {
    chunk_size = IREE_TOKENIZER_UNIGRAM_MIN_CHUNK_SIZE;
  }

  // Compute state size with trailing DP and pending buffers.
  // DP arrays: chunk_size + 1 entries (one chunk worth of DP state).
  // Pending buffer: chunk_size entries (max tokens per chunk when every
  // byte is a separate token).
  iree_host_size_t state_size = 0;
  iree_host_size_t best_score_offset = 0;
  iree_host_size_t best_token_id_offset = 0;
  iree_host_size_t best_length_offset = 0;
  iree_host_size_t pending_tokens_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          sizeof(iree_tokenizer_unigram_state_t), &state_size,
          IREE_STRUCT_FIELD(chunk_size + 1, float, &best_score_offset),
          IREE_STRUCT_FIELD(chunk_size + 1, int32_t, &best_token_id_offset),
          IREE_STRUCT_FIELD(chunk_size + 1, uint16_t, &best_length_offset),
          IREE_STRUCT_FIELD(chunk_size, iree_tokenizer_unigram_pending_token_t,
                            &pending_tokens_offset)));

  iree_tokenizer_unigram_model_t* model = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*model), (void**)&model));

  memset(model, 0, sizeof(*model));
  model->allocator = allocator;
  model->vocab = vocab;
  model->unk_token_id = unk_token_id;
  model->unk_score = unk_score;
  model->max_token_length = max_token_length;
  model->chunk_size = chunk_size;
  model->flags = flags;
  model->best_score_offset = best_score_offset;
  model->best_token_id_offset = best_token_id_offset;
  model->best_length_offset = best_length_offset;
  model->pending_tokens_offset = pending_tokens_offset;

  // Build byte_to_token fallback table.
  iree_tokenizer_unigram_build_byte_to_token(model);

  // Build prefix trie for token lookup.
  const iree_tokenizer_token_t* tokens = iree_tokenizer_vocab_tokens(vocab);
  iree_host_size_t token_count = iree_tokenizer_vocab_capacity(vocab);
  iree_const_byte_span_t string_table =
      iree_tokenizer_vocab_string_table(vocab);
  iree_status_t status = iree_tokenizer_vocab_trie_build(
      tokens, token_count, string_table, allocator, &model->trie);

  if (iree_status_is_ok(status)) {
    iree_tokenizer_model_initialize(&model->base,
                                    &iree_tokenizer_unigram_model_vtable,
                                    state_size, IREE_SV("Unigram"));
    *out_model = &model->base;
  } else {
    iree_tokenizer_unigram_model_destroy(&model->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_tokenizer_unigram_model_destroy(
    iree_tokenizer_model_t* base_model) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_unigram_model_t* model =
      (iree_tokenizer_unigram_model_t*)base_model;
  iree_allocator_t allocator = model->allocator;
  iree_tokenizer_vocab_trie_free(model->trie);
  iree_allocator_free(allocator, model);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// State Operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_unigram_state_initialize(
    const iree_tokenizer_model_t* base_model, void* storage,
    iree_tokenizer_model_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.model = base_model;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_unigram_state_deinitialize(
    iree_tokenizer_model_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Emits pending tokens to the output buffer. Returns the number of tokens
// emitted. Updates pending_emit_index.
static iree_host_size_t iree_tokenizer_unigram_emit_pending(
    iree_tokenizer_unigram_state_t* state,
    const iree_tokenizer_unigram_model_t* model,
    iree_tokenizer_token_output_t output, iree_host_size_t token_count) {
  iree_tokenizer_unigram_pending_token_t* pending =
      iree_tokenizer_unigram_state_pending_tokens(state, model);
  while (state->pending_emit_index < state->pending_count) {
    if (token_count >= output.capacity) break;
    iree_tokenizer_unigram_pending_token_t* token =
        &pending[state->pending_emit_index];
    output.token_ids[token_count] = token->token_id;
    if (output.token_offsets) {
      output.token_offsets[token_count].start =
          state->segment_base_offset + token->start_byte;
      output.token_offsets[token_count].end =
          state->segment_base_offset + token->end_byte;
    }
    ++token_count;
    ++state->pending_emit_index;
  }
  return token_count;
}

// Processes a single segment using chunked Viterbi. Handles both complete and
// partial segments. Returns the total token count (including previously emitted
// tokens from |token_count|). Sets |*out_segment_complete| to true when the
// segment has been fully processed and all tokens emitted.
static iree_status_t iree_tokenizer_unigram_process_segment(
    iree_tokenizer_unigram_state_t* state,
    const iree_tokenizer_unigram_model_t* model,
    iree_const_byte_span_t transform_buffer,
    const iree_tokenizer_segment_t* segment, bool is_partial,
    iree_tokenizer_token_output_t output, iree_host_size_t* in_out_token_count,
    bool* out_segment_complete) {
  iree_host_size_t token_count = *in_out_token_count;
  *out_segment_complete = false;

  // Validate segment bounds.
  if (segment->end > transform_buffer.data_length ||
      segment->start > segment->end) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "segment bounds out of range: [%" PRIhsz
                            ", %" PRIhsz ") in buffer of %" PRIhsz " bytes",
                            segment->start, segment->end,
                            transform_buffer.data_length);
  }

  const uint8_t* segment_data = transform_buffer.data + segment->start;
  iree_host_size_t segment_length = segment->end - segment->start;

  // Update segment base offset (may have changed due to reclaim).
  state->segment_base_offset = segment->start;

  // Resume emitting any pending tokens from a previous call.
  token_count =
      iree_tokenizer_unigram_emit_pending(state, model, output, token_count);
  if (state->pending_emit_index < state->pending_count) {
    // Output full, pending tokens remain.
    *in_out_token_count = token_count;
    return iree_ok_status();
  }

  // Pending buffer drained — update committed position.
  if (state->pending_count > 0) {
    state->committed_position = state->byte_position;
    state->pending_count = 0;
    state->pending_emit_index = 0;
  }

  // For partial segments, compute the safe processing boundary.
  // Hold back the last max_token_length bytes because tokens spanning the
  // segment boundary could be found if more data arrives.
  iree_host_size_t safe_end = segment_length;
  if (is_partial && segment_length > model->max_token_length) {
    safe_end = segment_length - model->max_token_length;
  } else if (is_partial) {
    safe_end = 0;  // Entire segment is within holdback zone.
  }

  // Process chunks until the segment is fully consumed or output fills.
  while (state->byte_position < segment_length) {
    if (token_count >= output.capacity) break;

    // For partial segments, stop before the holdback zone.
    if (is_partial && state->byte_position >= safe_end) break;

    // Determine chunk bounds. Use chunk_size for the Viterbi pass width.
    iree_host_size_t chunk_end = state->byte_position + model->chunk_size;
    if (chunk_end > segment_length) {
      chunk_end = segment_length;
    }
    // For partial segments, don't extend into the holdback zone.
    if (is_partial && chunk_end > safe_end) {
      chunk_end = safe_end;
    }

    // Align to UTF-8 character boundary if we're mid-segment.
    if (chunk_end < segment_length) {
      chunk_end = iree_tokenizer_unigram_align_utf8_boundary(
          segment_data, chunk_end, state->byte_position + 1);
      if (chunk_end <= state->byte_position) {
        // Degenerate: chunk too small to hold even one UTF-8 character.
        // Process whatever bytes are available.
        chunk_end = state->byte_position + 1;
      }
    }

    iree_host_size_t chunk_length = chunk_end - state->byte_position;
    const uint8_t* chunk_data = segment_data + state->byte_position;

    // Tokenize this chunk.
    iree_host_size_t subtokens = iree_tokenizer_unigram_tokenize_chunk(
        model, state, chunk_data, chunk_length, state->byte_position);

    if (subtokens == 0 && chunk_length > 0) {
      // Non-empty chunk that couldn't be tokenized — emit UNK if available.
      if (model->unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
        if (token_count >= output.capacity) break;
        output.token_ids[token_count] = model->unk_token_id;
        if (output.token_offsets) {
          output.token_offsets[token_count].start =
              state->segment_base_offset + state->byte_position;
          output.token_offsets[token_count].end =
              state->segment_base_offset + chunk_end;
        }
        ++token_count;
        state->byte_position = chunk_end;
        state->committed_position = chunk_end;
        continue;
      }
      // No UNK available and chunk is untokenizable — skip bytes to avoid
      // infinite loop. This should not happen with properly configured models.
      state->byte_position = chunk_end;
      state->committed_position = chunk_end;
      continue;
    }

    // Advance byte_position past the chunk (DP complete for these bytes).
    state->byte_position = chunk_end;

    // Set up pending tokens for emission.
    state->pending_count = subtokens;
    state->pending_emit_index = 0;

    // Emit as many as output capacity allows.
    token_count =
        iree_tokenizer_unigram_emit_pending(state, model, output, token_count);
    if (state->pending_emit_index < state->pending_count) {
      // Output full mid-chunk. committed_position stays at the previous chunk's
      // end (not yet fully emitted).
      break;
    }

    // All tokens emitted for this chunk.
    state->committed_position = state->byte_position;
    state->pending_count = 0;
    state->pending_emit_index = 0;
  }

  // Check if segment is fully processed and all tokens emitted.
  if (state->byte_position >= segment_length &&
      state->pending_emit_index >= state->pending_count) {
    // Reset streaming state for the next segment.
    state->byte_position = 0;
    state->committed_position = 0;
    state->pending_count = 0;
    state->pending_emit_index = 0;
    *out_segment_complete = true;
  }

  *in_out_token_count = token_count;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_unigram_state_encode(
    iree_tokenizer_model_state_t* base_state,
    iree_const_byte_span_t transform_buffer,
    iree_tokenizer_segment_list_t segments,
    iree_tokenizer_token_output_t output,
    iree_host_size_t* out_segments_consumed,
    iree_host_size_t* out_token_count) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)base_state;
  const iree_tokenizer_unigram_model_t* model =
      (const iree_tokenizer_unigram_model_t*)base_state->model;

  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  for (iree_host_size_t segment_index = 0; segment_index < segments.count;
       ++segment_index) {
    if (token_count >= output.capacity) break;

    bool is_partial =
        segments.last_is_partial && segment_index == segments.count - 1;
    bool segment_complete = false;

    IREE_RETURN_IF_ERROR(iree_tokenizer_unigram_process_segment(
        state, model, transform_buffer, &segments.values[segment_index],
        is_partial, output, &token_count, &segment_complete));

    if (segment_complete && !is_partial) {
      ++segments_consumed;
    } else if (!segment_complete) {
      // Segment not fully processed (output full or partial).
      // Stop processing further segments.
      break;
    }
    // For partial segments: segment_complete may be true (all available bytes
    // processed) but we don't count it as consumed — the pipeline will re-
    // present it with more data or call finalize.
  }

  *out_segments_consumed = segments_consumed;
  *out_token_count = token_count;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_unigram_state_finalize(
    iree_tokenizer_model_state_t* base_state,
    iree_tokenizer_token_output_t output, iree_host_size_t* out_token_count) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)base_state;
  const iree_tokenizer_unigram_model_t* model =
      (const iree_tokenizer_unigram_model_t*)base_state->model;

  // Drain any remaining pending tokens.
  iree_host_size_t token_count = 0;
  token_count =
      iree_tokenizer_unigram_emit_pending(state, model, output, token_count);

  if (state->pending_emit_index >= state->pending_count) {
    state->pending_count = 0;
    state->pending_emit_index = 0;
  }

  *out_token_count = token_count;
  return iree_ok_status();
}

static bool iree_tokenizer_unigram_state_has_pending(
    const iree_tokenizer_model_state_t* base_state) {
  const iree_tokenizer_unigram_state_t* state =
      (const iree_tokenizer_unigram_state_t*)base_state;
  return state->pending_emit_index < state->pending_count;
}

static iree_host_size_t iree_tokenizer_unigram_state_reclaim(
    iree_tokenizer_model_state_t* base_state) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)base_state;
  const iree_tokenizer_unigram_model_t* model =
      (const iree_tokenizer_unigram_model_t*)base_state->model;

  iree_host_size_t committed = state->committed_position;
  if (committed == 0) {
    return 0;
  }

  // Adjust byte_position relative to the new segment start.
  state->byte_position -= committed;
  state->committed_position = 0;

  // Adjust any remaining pending token offsets.
  if (state->pending_emit_index < state->pending_count) {
    iree_tokenizer_unigram_pending_token_t* pending =
        iree_tokenizer_unigram_state_pending_tokens(state, model);
    for (iree_host_size_t i = state->pending_emit_index;
         i < state->pending_count; ++i) {
      pending[i].start_byte -= committed;
      pending[i].end_byte -= committed;
    }
  }

  return committed;
}

static iree_status_t iree_tokenizer_unigram_get_token_string(
    const iree_tokenizer_model_t* base_model,
    iree_tokenizer_token_id_t token_id, iree_string_view_t* out_string) {
  const iree_tokenizer_unigram_model_t* model =
      (const iree_tokenizer_unigram_model_t*)base_model;
  *out_string = iree_tokenizer_vocab_token_text(model->vocab, token_id);
  if (out_string->size == 0 && token_id >= 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "token ID %" PRId32 " not in vocabulary", token_id);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_model_vtable_t iree_tokenizer_unigram_model_vtable =
    {
        .destroy = iree_tokenizer_unigram_model_destroy,
        .state_initialize = iree_tokenizer_unigram_state_initialize,
        .state_deinitialize = iree_tokenizer_unigram_state_deinitialize,
        .state_encode = iree_tokenizer_unigram_state_encode,
        .state_finalize = iree_tokenizer_unigram_state_finalize,
        .state_has_pending = iree_tokenizer_unigram_state_has_pending,
        .state_reclaim = iree_tokenizer_unigram_state_reclaim,
        .get_token_string = iree_tokenizer_unigram_get_token_string,
};
