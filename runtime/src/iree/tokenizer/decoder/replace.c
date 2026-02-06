// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/replace.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Replace Decoder Implementation
//===----------------------------------------------------------------------===//

// Maximum pattern/content length we support. Larger patterns are rare and
// would need heap allocation.
#define IREE_TOKENIZER_DECODER_REPLACE_MAX_PATTERN_LENGTH 16

typedef struct iree_tokenizer_decoder_replace_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
  // Pattern to search for (inline storage).
  char pattern[IREE_TOKENIZER_DECODER_REPLACE_MAX_PATTERN_LENGTH];
  iree_host_size_t pattern_length;
  // Content to replace with (inline storage).
  char content[IREE_TOKENIZER_DECODER_REPLACE_MAX_PATTERN_LENGTH];
  iree_host_size_t content_length;
} iree_tokenizer_decoder_replace_t;

typedef struct iree_tokenizer_decoder_replace_state_t {
  iree_tokenizer_decoder_state_t base;
  // Copied from decoder for cache locality.
  char pattern[IREE_TOKENIZER_DECODER_REPLACE_MAX_PATTERN_LENGTH];
  iree_host_size_t pattern_length;
  char content[IREE_TOKENIZER_DECODER_REPLACE_MAX_PATTERN_LENGTH];
  iree_host_size_t content_length;
  // Position within current token when resuming after buffer-full.
  // When non-zero, the first token in the next process() call should resume
  // from this position rather than reprocessing from the beginning.
  iree_host_size_t resume_position;
} iree_tokenizer_decoder_replace_state_t;

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_replace_vtable;

iree_status_t iree_tokenizer_decoder_replace_allocate(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;

  if (pattern.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Replace pattern must not be empty");
  }
  if (pattern.size < content.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Replace pattern length (%zu) must be >= content "
                            "length (%zu) for shrinking decoder",
                            pattern.size, content.size);
  }
  if (pattern.size > IREE_TOKENIZER_DECODER_REPLACE_MAX_PATTERN_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Replace pattern length (%zu) exceeds maximum (%d)",
                            pattern.size,
                            IREE_TOKENIZER_DECODER_REPLACE_MAX_PATTERN_LENGTH);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_decoder_replace_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  iree_tokenizer_decoder_initialize(
      &decoder->base, &iree_tokenizer_decoder_replace_vtable,
      sizeof(iree_tokenizer_decoder_replace_state_t),
      IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS);
  decoder->allocator = allocator;

  memcpy(decoder->pattern, pattern.data, pattern.size);
  decoder->pattern_length = pattern.size;
  memcpy(decoder->content, content.data, content.size);
  decoder->content_length = content.size;

  *out_decoder = &decoder->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_replace_destroy(
    iree_tokenizer_decoder_t* base_decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_replace_t* decoder =
      (iree_tokenizer_decoder_replace_t*)base_decoder;
  iree_allocator_t allocator = decoder->allocator;
  iree_allocator_free(allocator, decoder);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_replace_state_initialize(
    const iree_tokenizer_decoder_t* base_decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_decoder_replace_t* decoder =
      (const iree_tokenizer_decoder_replace_t*)base_decoder;
  iree_tokenizer_decoder_replace_state_t* state =
      (iree_tokenizer_decoder_replace_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.decoder = base_decoder;

  // Copy config to state for cache locality.
  memcpy(state->pattern, decoder->pattern, decoder->pattern_length);
  state->pattern_length = decoder->pattern_length;
  memcpy(state->content, decoder->content, decoder->content_length);
  state->content_length = decoder->content_length;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_replace_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_replace_state_process(
    iree_tokenizer_decoder_state_t* base_state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  iree_tokenizer_decoder_replace_state_t* state =
      (iree_tokenizer_decoder_replace_state_t*)base_state;

  const char* pattern = state->pattern;
  const iree_host_size_t pattern_length = state->pattern_length;
  const char* content = state->content;
  const iree_host_size_t content_length = state->content_length;

  iree_host_size_t strings_consumed = 0;
  iree_host_size_t write_position = 0;

  for (iree_host_size_t i = 0; i < token_strings.count; ++i) {
    iree_string_view_t token = token_strings.values[i];

    // Resume from saved position if this is the first token after buffer-full.
    iree_host_size_t read_position = (i == 0) ? state->resume_position : 0;

    while (read_position < token.size) {
      // Search for pattern starting at read_position.
      const char* match = NULL;
      for (iree_host_size_t j = read_position; j + pattern_length <= token.size;
           ++j) {
        if (memcmp(token.data + j, pattern, pattern_length) == 0) {
          match = token.data + j;
          break;
        }
      }

      if (match) {
        // Copy everything before the match.
        iree_host_size_t prefix_length = match - (token.data + read_position);
        if (write_position + prefix_length > output.size) {
          // Buffer full - can't fit prefix. Save position for resume.
          state->resume_position = read_position;
          *out_strings_consumed = strings_consumed;
          *out_bytes_written = write_position;
          return iree_ok_status();
        }
        // Use memmove - source and destination may overlap when used in
        // sequence decoder (same buffer for input and output).
        if (prefix_length > 0) {
          memmove(output.data + write_position, token.data + read_position,
                  prefix_length);
          write_position += prefix_length;
        }
        read_position += prefix_length;

        // Write replacement content.
        if (write_position + content_length > output.size) {
          // Buffer full - can't fit replacement. Save position for resume.
          state->resume_position = read_position;
          *out_strings_consumed = strings_consumed;
          *out_bytes_written = write_position;
          return iree_ok_status();
        }
        if (content_length > 0) {
          memcpy(output.data + write_position, content, content_length);
          write_position += content_length;
        }
        read_position += pattern_length;
      } else {
        // No more matches - copy remainder.
        iree_host_size_t remainder_length = token.size - read_position;
        if (write_position + remainder_length > output.size) {
          // Buffer full - cannot fit remainder. Save position for resume.
          state->resume_position = read_position;
          *out_strings_consumed = strings_consumed;
          *out_bytes_written = write_position;
          return iree_ok_status();
        }
        // Use memmove - source and destination may overlap when used in
        // sequence decoder (same buffer for input and output).
        memmove(output.data + write_position, token.data + read_position,
                remainder_length);
        write_position += remainder_length;
        read_position = token.size;
      }
    }
    // Token fully processed - clear resume position and count as consumed.
    state->resume_position = 0;
    strings_consumed++;
  }

  *out_strings_consumed = strings_consumed;
  *out_bytes_written = write_position;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_replace_state_finalize(
    iree_tokenizer_decoder_state_t* state, iree_mutable_string_view_t output,
    iree_host_size_t* out_written) {
  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_replace_state_has_pending(
    const iree_tokenizer_decoder_state_t* state) {
  return false;
}

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_replace_vtable = {
        .destroy = iree_tokenizer_decoder_replace_destroy,
        .state_initialize = iree_tokenizer_decoder_replace_state_initialize,
        .state_deinitialize = iree_tokenizer_decoder_replace_state_deinitialize,
        .state_process = iree_tokenizer_decoder_replace_state_process,
        .state_finalize = iree_tokenizer_decoder_replace_state_finalize,
        .state_has_pending = iree_tokenizer_decoder_replace_state_has_pending,
        .flags = IREE_TOKENIZER_DECODER_FLAG_SHRINKING,
};
