// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WordPiece tokenization using greedy longest-match with continuation prefix.
//
// ALGORITHM:
// For each segment (word):
//   1. Count Unicode characters; if > max_input_chars_per_word, emit [UNK]
//   2. Starting at byte 0, try the longest substring in vocab (full → shorter)
//   3. For positions > 0, prepend continuing_subword_prefix before lookup
//   4. If any position has no match, the ENTIRE word becomes [UNK]
//   5. Otherwise, emit all matched sub-tokens
//
// PRE-COMPUTATION:
// Because a failure at any position invalidates all prior sub-tokens for that
// word, we pre-compute all sub-tokens into a trailing buffer before emitting.
// This avoids partial emission followed by retraction.
//
// COMPLEXITY:
// - Time: O(n * L) per word, where n = word length, L = max token length
//   (each position tries at most L substrings)
// - Memory: O(max_input_chars_per_word) for the pending token buffer

#include "iree/tokenizer/model/wordpiece.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Maximum buffer size for vocabulary lookup candidates (prefix + substring).
// BERT vocabs rarely have tokens longer than 30 bytes; this accommodates
// max_token_length + prefix_length generously.
#define IREE_TOKENIZER_WORDPIECE_MAX_LOOKUP_BYTES 512

//===----------------------------------------------------------------------===//
// Pending Token Entry
//===----------------------------------------------------------------------===//

// Pre-computed sub-token for deferred emission.
typedef struct iree_tokenizer_wordpiece_pending_token_t {
  iree_tokenizer_token_id_t token_id;
  // Byte offsets relative to the transform buffer.
  iree_host_size_t start_byte;
  iree_host_size_t end_byte;
} iree_tokenizer_wordpiece_pending_token_t;

//===----------------------------------------------------------------------===//
// WordPiece Model Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_wordpiece_model_t {
  iree_tokenizer_model_t base;
  iree_allocator_t allocator;
  const iree_tokenizer_vocab_t* vocab;  // NOT owned.

  // UNK token ID resolved from vocabulary special tokens.
  iree_tokenizer_token_id_t unk_token_id;

  // Prefix prepended to non-initial subwords (e.g., "##").
  char continuing_subword_prefix[16];
  iree_host_size_t continuing_subword_prefix_length;

  // Maximum Unicode characters per word before falling back to [UNK].
  iree_host_size_t max_input_chars_per_word;

  // Offset into state slab for the pending tokens buffer.
  iree_host_size_t pending_tokens_offset;
} iree_tokenizer_wordpiece_model_t;

//===----------------------------------------------------------------------===//
// WordPiece State Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_wordpiece_state_t {
  iree_tokenizer_model_state_t base;

  // Pre-computed tokens for the current segment.
  iree_host_size_t pending_count;
  iree_host_size_t pending_emit_index;

  // Trailing buffer (accessed via model->pending_tokens_offset):
  //   iree_tokenizer_wordpiece_pending_token_t
  //       pending_tokens[max_input_chars_per_word]
} iree_tokenizer_wordpiece_state_t;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Access the trailing pending tokens buffer.
static inline iree_tokenizer_wordpiece_pending_token_t*
iree_tokenizer_wordpiece_state_pending_tokens(
    iree_tokenizer_wordpiece_state_t* state,
    const iree_tokenizer_wordpiece_model_t* model) {
  return (
      iree_tokenizer_wordpiece_pending_token_t*)((uint8_t*)state +
                                                 model->pending_tokens_offset);
}

// Returns the byte length of a UTF-8 sequence starting at |byte|.
// Returns 1 for invalid lead bytes (treats as single-byte character).
static inline iree_host_size_t iree_tokenizer_wordpiece_utf8_char_length(
    uint8_t byte) {
  if (byte < 0x80) return 1;
  if ((byte & 0xE0) == 0xC0) return 2;
  if ((byte & 0xF0) == 0xE0) return 3;
  if ((byte & 0xF8) == 0xF0) return 4;
  return 1;  // Invalid lead byte, treat as single byte.
}

// Counts Unicode characters in a byte range.
static iree_host_size_t iree_tokenizer_wordpiece_count_chars(
    const uint8_t* data, iree_host_size_t length) {
  iree_host_size_t count = 0;
  iree_host_size_t position = 0;
  while (position < length) {
    position += iree_tokenizer_wordpiece_utf8_char_length(data[position]);
    ++count;
  }
  return count;
}

// Backs up one UTF-8 character from |position| in |data|.
// Returns the byte offset of the previous character's start.
// Precondition: position > 0.
static iree_host_size_t iree_tokenizer_wordpiece_prev_char(
    const uint8_t* data, iree_host_size_t position) {
  // Walk backwards past continuation bytes (10xxxxxx).
  iree_host_size_t p = position - 1;
  while (p > 0 && (data[p] & 0xC0) == 0x80) {
    --p;
  }
  return p;
}

// Attempts to find the longest vocabulary match starting at |start_byte|
// within the segment |data[0..segment_length)|. For non-initial positions
// (start_byte > 0), the continuing_subword_prefix is prepended before lookup.
//
// Returns the token ID and sets |out_end_byte| to the end of the match.
// Returns IREE_TOKENIZER_TOKEN_ID_INVALID if no match found.
static iree_tokenizer_token_id_t iree_tokenizer_wordpiece_find_longest_match(
    const iree_tokenizer_wordpiece_model_t* model, const uint8_t* data,
    iree_host_size_t segment_length, iree_host_size_t start_byte,
    iree_host_size_t* out_end_byte) {
  char lookup_buffer[IREE_TOKENIZER_WORDPIECE_MAX_LOOKUP_BYTES];
  const iree_host_size_t prefix_length =
      (start_byte > 0) ? model->continuing_subword_prefix_length : 0;

  // Copy prefix into lookup buffer for non-initial positions.
  if (prefix_length > 0) {
    memcpy(lookup_buffer, model->continuing_subword_prefix, prefix_length);
  }

  // Try decreasing lengths from end of segment down to start_byte + 1 char.
  iree_host_size_t end = segment_length;
  while (end > start_byte) {
    iree_host_size_t substring_length = end - start_byte;
    iree_host_size_t total_length = prefix_length + substring_length;

    // Skip if candidate exceeds lookup buffer.
    if (total_length <= sizeof(lookup_buffer)) {
      memcpy(lookup_buffer + prefix_length, data + start_byte,
             substring_length);
      iree_tokenizer_token_id_t token_id = iree_tokenizer_vocab_lookup(
          model->vocab, iree_make_string_view(lookup_buffer, total_length));
      if (token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
        *out_end_byte = end;
        return token_id;
      }
    }

    // Shorten by one character from the right.
    end = iree_tokenizer_wordpiece_prev_char(data, end);
  }

  return IREE_TOKENIZER_TOKEN_ID_INVALID;
}

// Pre-computes all sub-tokens for a single segment into the pending buffer.
// Returns the number of tokens written to pending, or 0 if the word is "bad"
// (in which case the caller should emit [UNK]).
static iree_host_size_t iree_tokenizer_wordpiece_tokenize_segment(
    const iree_tokenizer_wordpiece_model_t* model,
    iree_tokenizer_wordpiece_state_t* state, const uint8_t* segment_data,
    iree_host_size_t segment_length, iree_host_size_t segment_base_offset) {
  iree_tokenizer_wordpiece_pending_token_t* pending_tokens =
      iree_tokenizer_wordpiece_state_pending_tokens(state, model);

  // Empty segments produce no tokens.
  if (segment_length == 0) return 0;

  // Check character count limit.
  iree_host_size_t char_count =
      iree_tokenizer_wordpiece_count_chars(segment_data, segment_length);
  if (char_count > model->max_input_chars_per_word) {
    // Word too long — will be replaced with [UNK] by caller.
    return 0;
  }

  // Greedy longest-match from left to right.
  iree_host_size_t token_count = 0;
  iree_host_size_t position = 0;
  while (position < segment_length) {
    iree_host_size_t end_byte = 0;
    iree_tokenizer_token_id_t token_id =
        iree_tokenizer_wordpiece_find_longest_match(
            model, segment_data, segment_length, position, &end_byte);

    if (token_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
      // No match found at this position — entire word is bad.
      return 0;
    }

    // Ensure we don't overflow the pending tokens buffer. With malformed UTF-8,
    // the character count may underestimate tokens (e.g., 0xF1 bytes count as
    // 4-byte sequences but vocab may have single-byte tokens). Treat as bad.
    if (token_count >= model->max_input_chars_per_word) {
      return 0;
    }

    // Store the pre-computed token.
    pending_tokens[token_count].token_id = token_id;
    pending_tokens[token_count].start_byte = segment_base_offset + position;
    pending_tokens[token_count].end_byte = segment_base_offset + end_byte;
    ++token_count;

    position = end_byte;
  }

  return token_count;
}

//===----------------------------------------------------------------------===//
// WordPiece Model Allocation
//===----------------------------------------------------------------------===//

static const iree_tokenizer_model_vtable_t
    iree_tokenizer_wordpiece_model_vtable;
static void iree_tokenizer_wordpiece_model_destroy(
    iree_tokenizer_model_t* base_model);

iree_status_t iree_tokenizer_wordpiece_model_allocate(
    const iree_tokenizer_vocab_t* vocab,
    iree_string_view_t continuing_subword_prefix,
    iree_host_size_t max_input_chars_per_word,
    iree_tokenizer_wordpiece_flags_t flags, iree_allocator_t allocator,
    iree_tokenizer_model_t** out_model) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_model);
  *out_model = NULL;

  if (continuing_subword_prefix.size >
      sizeof(
          ((iree_tokenizer_wordpiece_model_t*)0)->continuing_subword_prefix)) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "continuing_subword_prefix too long (%" PRIhsz
                             " bytes, max %zu)",
                             continuing_subword_prefix.size,
                             sizeof(((iree_tokenizer_wordpiece_model_t*)0)
                                        ->continuing_subword_prefix)));
  }
  if (max_input_chars_per_word == 0) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "max_input_chars_per_word must be > 0"));
  }

  // Resolve UNK token from vocabulary.
  iree_tokenizer_special_ids_t special_ids =
      iree_tokenizer_vocab_special_ids(vocab);
  iree_tokenizer_token_id_t unk_token_id = special_ids.unk;
  if (unk_token_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "WordPiece model requires [UNK] token in "
                             "vocabulary (none found via special_ids)"));
  }

  // Compute state size with trailing pending tokens buffer.
  iree_host_size_t state_size = 0;
  iree_host_size_t pending_tokens_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_tokenizer_wordpiece_state_t), &state_size,
              IREE_STRUCT_FIELD(max_input_chars_per_word,
                                iree_tokenizer_wordpiece_pending_token_t,
                                &pending_tokens_offset)));

  iree_tokenizer_wordpiece_model_t* model = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*model), (void**)&model));

  memset(model, 0, sizeof(*model));
  model->allocator = allocator;
  model->vocab = vocab;
  model->unk_token_id = unk_token_id;
  model->max_input_chars_per_word = max_input_chars_per_word;
  model->pending_tokens_offset = pending_tokens_offset;

  // Store prefix.
  memcpy(model->continuing_subword_prefix, continuing_subword_prefix.data,
         continuing_subword_prefix.size);
  model->continuing_subword_prefix_length = continuing_subword_prefix.size;

  iree_tokenizer_model_initialize(&model->base,
                                  &iree_tokenizer_wordpiece_model_vtable,
                                  state_size, IREE_SV("WordPiece"));
  *out_model = &model->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_wordpiece_model_destroy(
    iree_tokenizer_model_t* base_model) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_wordpiece_model_t* model =
      (iree_tokenizer_wordpiece_model_t*)base_model;
  iree_allocator_t allocator = model->allocator;
  iree_allocator_free(allocator, model);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// WordPiece State Operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_wordpiece_state_initialize(
    const iree_tokenizer_model_t* base_model, void* storage,
    iree_tokenizer_model_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_wordpiece_state_t* state =
      (iree_tokenizer_wordpiece_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.model = base_model;
  state->pending_count = 0;
  state->pending_emit_index = 0;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_wordpiece_state_deinitialize(
    iree_tokenizer_model_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_wordpiece_state_encode(
    iree_tokenizer_model_state_t* base_state,
    iree_const_byte_span_t transform_buffer,
    iree_tokenizer_segment_list_t segments,
    iree_tokenizer_token_output_t output,
    iree_host_size_t* out_segments_consumed,
    iree_host_size_t* out_token_count) {
  iree_tokenizer_wordpiece_state_t* state =
      (iree_tokenizer_wordpiece_state_t*)base_state;
  const iree_tokenizer_wordpiece_model_t* model =
      (const iree_tokenizer_wordpiece_model_t*)base_state->model;

  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;
  iree_host_size_t segment_index = 0;

  // Determine how many segments to process (skip last if partial).
  iree_host_size_t processable_count = segments.count;
  if (segments.last_is_partial && processable_count > 0) {
    --processable_count;
  }

  // Resume emitting any pending tokens from a previous call that filled output.
  // The segment that produced these tokens is always segment 0 of the current
  // list (the caller did not advance past it since segments_consumed was 0).
  while (state->pending_emit_index < state->pending_count) {
    if (token_count >= output.capacity) {
      *out_segments_consumed = segments_consumed;
      *out_token_count = token_count;
      return iree_ok_status();
    }
    iree_tokenizer_wordpiece_pending_token_t* pending =
        iree_tokenizer_wordpiece_state_pending_tokens(state, model);
    iree_tokenizer_wordpiece_pending_token_t* token =
        &pending[state->pending_emit_index];
    output.token_ids[token_count] = token->token_id;
    if (output.token_offsets) {
      output.token_offsets[token_count].start = token->start_byte;
      output.token_offsets[token_count].end = token->end_byte;
    }
    ++token_count;
    ++state->pending_emit_index;
  }

  // If we just finished emitting pending tokens, advance past segment 0.
  if (state->pending_emit_index > 0 &&
      state->pending_emit_index >= state->pending_count) {
    ++segments_consumed;
    ++segment_index;
    state->pending_count = 0;
    state->pending_emit_index = 0;
  }

  // Process remaining segments.
  while (segment_index < processable_count) {
    if (token_count >= output.capacity) {
      *out_segments_consumed = segments_consumed;
      *out_token_count = token_count;
      return iree_ok_status();
    }

    const iree_tokenizer_segment_t* segment = &segments.values[segment_index];

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

    // Pre-compute all sub-tokens for this segment.
    iree_host_size_t subtokens = iree_tokenizer_wordpiece_tokenize_segment(
        model, state, segment_data, segment_length, segment->start);

    if (subtokens == 0 && segment_length > 0) {
      // Word is "bad" or too long — emit [UNK].
      output.token_ids[token_count] = model->unk_token_id;
      if (output.token_offsets) {
        output.token_offsets[token_count].start = segment->start;
        output.token_offsets[token_count].end = segment->end;
      }
      ++token_count;
      ++segments_consumed;
      ++segment_index;
      continue;
    }

    if (subtokens == 0) {
      // Empty segment — skip.
      ++segments_consumed;
      ++segment_index;
      continue;
    }

    // Set up pending tokens for emission.
    state->pending_count = subtokens;
    state->pending_emit_index = 0;

    // Emit as many as output capacity allows.
    iree_tokenizer_wordpiece_pending_token_t* pending =
        iree_tokenizer_wordpiece_state_pending_tokens(state, model);
    while (state->pending_emit_index < state->pending_count) {
      if (token_count >= output.capacity) {
        *out_segments_consumed = segments_consumed;
        *out_token_count = token_count;
        return iree_ok_status();
      }
      iree_tokenizer_wordpiece_pending_token_t* token =
          &pending[state->pending_emit_index];
      output.token_ids[token_count] = token->token_id;
      if (output.token_offsets) {
        output.token_offsets[token_count].start = token->start_byte;
        output.token_offsets[token_count].end = token->end_byte;
      }
      ++token_count;
      ++state->pending_emit_index;
    }

    // Fully emitted this segment.
    state->pending_count = 0;
    state->pending_emit_index = 0;
    ++segments_consumed;
    ++segment_index;
  }

  *out_segments_consumed = segments_consumed;
  *out_token_count = token_count;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_wordpiece_state_finalize(
    iree_tokenizer_model_state_t* base_state,
    iree_tokenizer_token_output_t output, iree_host_size_t* out_token_count) {
  // WordPiece processes segments completely in state_encode.
  // Finalize only needs to emit remaining pending tokens (if output filled
  // mid-segment during the last encode call).
  iree_tokenizer_wordpiece_state_t* state =
      (iree_tokenizer_wordpiece_state_t*)base_state;
  const iree_tokenizer_wordpiece_model_t* model =
      (const iree_tokenizer_wordpiece_model_t*)base_state->model;

  iree_host_size_t token_count = 0;
  while (state->pending_emit_index < state->pending_count) {
    if (token_count >= output.capacity) break;
    iree_tokenizer_wordpiece_pending_token_t* pending =
        iree_tokenizer_wordpiece_state_pending_tokens(state, model);
    iree_tokenizer_wordpiece_pending_token_t* token =
        &pending[state->pending_emit_index];
    output.token_ids[token_count] = token->token_id;
    if (output.token_offsets) {
      output.token_offsets[token_count].start = token->start_byte;
      output.token_offsets[token_count].end = token->end_byte;
    }
    ++token_count;
    ++state->pending_emit_index;
  }

  if (state->pending_emit_index >= state->pending_count) {
    state->pending_count = 0;
    state->pending_emit_index = 0;
  }

  *out_token_count = token_count;
  return iree_ok_status();
}

static bool iree_tokenizer_wordpiece_state_has_pending(
    const iree_tokenizer_model_state_t* base_state) {
  const iree_tokenizer_wordpiece_state_t* state =
      (const iree_tokenizer_wordpiece_state_t*)base_state;
  return state->pending_emit_index < state->pending_count;
}

static iree_host_size_t iree_tokenizer_wordpiece_state_reclaim(
    iree_tokenizer_model_state_t* state) {
  // WordPiece processes segments independently; no partial segment state
  // to reclaim.
  return 0;
}

static iree_status_t iree_tokenizer_wordpiece_get_token_string(
    const iree_tokenizer_model_t* base_model,
    iree_tokenizer_token_id_t token_id, iree_string_view_t* out_string) {
  const iree_tokenizer_wordpiece_model_t* model =
      (const iree_tokenizer_wordpiece_model_t*)base_model;
  *out_string = iree_tokenizer_vocab_token_text(model->vocab, token_id);
  if (out_string->size == 0 && token_id >= 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "token ID %" PRId32 " not in vocabulary", token_id);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// WordPiece VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_model_vtable_t
    iree_tokenizer_wordpiece_model_vtable = {
        .destroy = iree_tokenizer_wordpiece_model_destroy,
        .state_initialize = iree_tokenizer_wordpiece_state_initialize,
        .state_deinitialize = iree_tokenizer_wordpiece_state_deinitialize,
        .state_encode = iree_tokenizer_wordpiece_state_encode,
        .state_finalize = iree_tokenizer_wordpiece_state_finalize,
        .state_has_pending = iree_tokenizer_wordpiece_state_has_pending,
        .state_reclaim = iree_tokenizer_wordpiece_state_reclaim,
        .get_token_string = iree_tokenizer_wordpiece_get_token_string,
};
