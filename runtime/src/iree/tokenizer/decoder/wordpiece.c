// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/wordpiece.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Cleanup Pattern Table
//===----------------------------------------------------------------------===//

// Cleanup patterns sorted by length DESCENDING for longest-match-first.
// These patterns fix common tokenization artifacts when applied to the
// LOGICAL token (after space prepending).
typedef struct iree_tokenizer_wordpiece_cleanup_pattern_t {
  const char* pattern;
  const char* replacement;
  uint8_t pattern_length;
  uint8_t replacement_length;
} iree_tokenizer_wordpiece_cleanup_pattern_t;

static const iree_tokenizer_wordpiece_cleanup_pattern_t
    kWordPieceCleanupPatterns[] = {
        // Longest patterns first for correct matching.
        {" do not", " don't", 8, 7}, {" n't", "n't", 4, 3},
        {" 've", "'ve", 4, 3},       {" 're", "'re", 4, 3},
        {" ' ", "'", 3, 1},          {" 'm", "'m", 3, 2},
        {" 's", "'s", 3, 2},         {" .", ".", 2, 1},
        {" ?", "?", 2, 1},           {" !", "!", 2, 1},
        {" ,", ",", 2, 1},
};

#define IREE_TOKENIZER_WORDPIECE_CLEANUP_PATTERN_COUNT \
  (sizeof(kWordPieceCleanupPatterns) / sizeof(kWordPieceCleanupPatterns[0]))

//===----------------------------------------------------------------------===//
// WordPiece Decoder Implementation
//===----------------------------------------------------------------------===//

// Processing phases for streaming support.
typedef enum iree_tokenizer_decoder_wordpiece_phase_e {
  // Need to handle prefix/space/cleanup for current token.
  IREE_TOKENIZER_DECODER_WORDPIECE_PHASE_TOKEN_START = 0,
  // Prefix/space/cleanup done, copying remaining token bytes.
  IREE_TOKENIZER_DECODER_WORDPIECE_PHASE_COPY_BYTES = 1,
} iree_tokenizer_decoder_wordpiece_phase_t;

typedef struct iree_tokenizer_decoder_wordpiece_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
  char prefix[IREE_TOKENIZER_WORDPIECE_MAX_PREFIX_LENGTH];
  uint8_t prefix_length;
  bool cleanup;
} iree_tokenizer_decoder_wordpiece_t;

typedef struct iree_tokenizer_decoder_wordpiece_state_t {
  iree_tokenizer_decoder_state_t base;
  // Config copied for cache locality during hot path.
  char prefix[IREE_TOKENIZER_WORDPIECE_MAX_PREFIX_LENGTH];
  uint8_t prefix_length;
  bool cleanup;
  // Per-stream state.
  bool is_first_token;
  // Per-token state (reset when token fully consumed).
  iree_tokenizer_decoder_wordpiece_phase_t phase;
  iree_host_size_t copy_position;
} iree_tokenizer_decoder_wordpiece_state_t;

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_wordpiece_vtable;

iree_status_t iree_tokenizer_decoder_wordpiece_allocate(
    iree_tokenizer_decoder_wordpiece_config_t config,
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;

  // Validate prefix length.
  if (config.prefix.size > IREE_TOKENIZER_WORDPIECE_MAX_PREFIX_LENGTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "WordPiece prefix too long: %" PRIhsz " bytes (max %d)",
        config.prefix.size, IREE_TOKENIZER_WORDPIECE_MAX_PREFIX_LENGTH);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_decoder_wordpiece_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  iree_tokenizer_decoder_initialize(
      &decoder->base, &iree_tokenizer_decoder_wordpiece_vtable,
      sizeof(iree_tokenizer_decoder_wordpiece_state_t),
      IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS |
          IREE_TOKENIZER_DECODER_CAPABILITY_POSITION_SENSITIVE);
  decoder->allocator = allocator;

  // Store prefix.
  if (config.prefix.size > 0) {
    memcpy(decoder->prefix, config.prefix.data, config.prefix.size);
  }
  decoder->prefix_length = (uint8_t)config.prefix.size;
  decoder->cleanup = config.cleanup;

  *out_decoder = &decoder->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_wordpiece_destroy(
    iree_tokenizer_decoder_t* base_decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_wordpiece_t* decoder =
      (iree_tokenizer_decoder_wordpiece_t*)base_decoder;
  iree_allocator_t allocator = decoder->allocator;
  iree_allocator_free(allocator, decoder);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_wordpiece_state_initialize(
    const iree_tokenizer_decoder_t* base_decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_decoder_wordpiece_t* decoder =
      (const iree_tokenizer_decoder_wordpiece_t*)base_decoder;
  iree_tokenizer_decoder_wordpiece_state_t* state =
      (iree_tokenizer_decoder_wordpiece_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.decoder = base_decoder;

  // Copy config for cache-friendly hot path.
  memcpy(state->prefix, decoder->prefix, decoder->prefix_length);
  state->prefix_length = decoder->prefix_length;
  state->cleanup = decoder->cleanup;

  // Initialize per-stream state.
  state->is_first_token = true;
  state->phase = IREE_TOKENIZER_DECODER_WORDPIECE_PHASE_TOKEN_START;
  state->copy_position = 0;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_wordpiece_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Processes a single token through the WordPiece decoder.
// Returns true if token was fully consumed, false if buffer full (resume
// later).
static bool iree_tokenizer_decoder_wordpiece_process_token(
    iree_tokenizer_decoder_wordpiece_state_t* state, iree_string_view_t token,
    iree_mutable_string_view_t output, iree_host_size_t* write_position) {
  // Phase 1: TOKEN_START - Determine prefix/space, apply cleanup.
  if (state->phase == IREE_TOKENIZER_DECODER_WORDPIECE_PHASE_TOKEN_START) {
    bool need_space = false;
    iree_host_size_t effective_start = 0;

    // For non-first tokens, determine if we strip prefix or prepend space.
    if (!state->is_first_token) {
      if (token.size >= state->prefix_length && state->prefix_length > 0 &&
          memcmp(token.data, state->prefix, state->prefix_length) == 0) {
        // Token has continuation prefix (e.g., "##") - strip it.
        effective_start = state->prefix_length;
      } else {
        // Token needs space separator.
        need_space = true;
      }
    }

    // Build virtual window for cleanup pattern matching.
    // Window represents the LOGICAL token: [virtual_space?] + token_bytes
    uint8_t window[8];
    iree_host_size_t window_length = 0;

    if (need_space) {
      window[window_length++] = ' ';
    }

    iree_host_size_t token_remaining = token.size - effective_start;
    iree_host_size_t bytes_to_copy =
        (token_remaining < sizeof(window) - window_length)
            ? token_remaining
            : sizeof(window) - window_length;
    if (bytes_to_copy > 0) {
      memcpy(window + window_length, token.data + effective_start,
             bytes_to_copy);
      window_length += bytes_to_copy;
    }

    // Try cleanup patterns (longest-first) if cleanup is enabled.
    bool pattern_matched = false;
    if (state->cleanup && need_space) {
      for (iree_host_size_t i = 0;
           i < IREE_TOKENIZER_WORDPIECE_CLEANUP_PATTERN_COUNT; ++i) {
        const iree_tokenizer_wordpiece_cleanup_pattern_t* p =
            &kWordPieceCleanupPatterns[i];
        if (window_length >= p->pattern_length &&
            memcmp(window, p->pattern, p->pattern_length) == 0) {
          // ATOMIC CHECK: Ensure replacement fits before committing.
          if (*write_position + p->replacement_length > output.size) {
            return false;  // Buffer full, retry entire token start.
          }
          // Write replacement.
          memcpy(output.data + *write_position, p->replacement,
                 p->replacement_length);
          *write_position += p->replacement_length;
          // Advance past matched portion in token (subtract 1 for virtual
          // space).
          effective_start += p->pattern_length - 1;
          pattern_matched = true;
          need_space = false;  // Space consumed by pattern.
          break;
        }
      }
    }

    // If no pattern matched and we need space, emit it.
    if (need_space && !pattern_matched) {
      if (*write_position >= output.size) {
        return false;  // Buffer full, retry entire token start.
      }
      output.data[(*write_position)++] = ' ';
    }

    // Transition to COPY_BYTES phase.
    state->phase = IREE_TOKENIZER_DECODER_WORDPIECE_PHASE_COPY_BYTES;
    state->copy_position = effective_start;
  }

  // Phase 2: COPY_BYTES - Copy remaining token bytes.
  while (state->copy_position < token.size) {
    if (*write_position >= output.size) {
      return false;  // Buffer full, will resume at copy_position.
    }
    output.data[(*write_position)++] = token.data[state->copy_position++];
  }

  // Token fully consumed - reset for next token.
  state->is_first_token = false;
  state->phase = IREE_TOKENIZER_DECODER_WORDPIECE_PHASE_TOKEN_START;
  state->copy_position = 0;
  return true;
}

static iree_status_t iree_tokenizer_decoder_wordpiece_state_process(
    iree_tokenizer_decoder_state_t* base_state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  iree_tokenizer_decoder_wordpiece_state_t* state =
      (iree_tokenizer_decoder_wordpiece_state_t*)base_state;

  iree_host_size_t strings_consumed = 0;
  iree_host_size_t write_position = 0;

  for (iree_host_size_t i = 0; i < token_strings.count; ++i) {
    iree_string_view_t token = token_strings.values[i];

    bool token_consumed = iree_tokenizer_decoder_wordpiece_process_token(
        state, token, output, &write_position);

    if (token_consumed) {
      strings_consumed++;
    } else {
      break;  // Buffer full mid-token.
    }
  }

  *out_strings_consumed = strings_consumed;
  *out_bytes_written = write_position;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_wordpiece_state_finalize(
    iree_tokenizer_decoder_state_t* state, iree_mutable_string_view_t output,
    iree_host_size_t* out_written) {
  // WordPiece decoder has no pending state to flush.
  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_wordpiece_state_has_pending(
    const iree_tokenizer_decoder_state_t* base_state) {
  const iree_tokenizer_decoder_wordpiece_state_t* state =
      (const iree_tokenizer_decoder_wordpiece_state_t*)base_state;
  // Pending if we're mid-token (in COPY_BYTES phase with progress).
  return state->phase == IREE_TOKENIZER_DECODER_WORDPIECE_PHASE_COPY_BYTES;
}

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_wordpiece_vtable = {
        .destroy = iree_tokenizer_decoder_wordpiece_destroy,
        .state_initialize = iree_tokenizer_decoder_wordpiece_state_initialize,
        .state_deinitialize =
            iree_tokenizer_decoder_wordpiece_state_deinitialize,
        .state_process = iree_tokenizer_decoder_wordpiece_state_process,
        .state_finalize = iree_tokenizer_decoder_wordpiece_state_finalize,
        .state_has_pending = iree_tokenizer_decoder_wordpiece_state_has_pending,
        // No SHRINKING flag — WordPiece can expand output (space prepend).
        .flags = 0,
};
