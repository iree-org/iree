// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/strip.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Strip Decoder Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_decoder_strip_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
  // Content to strip (inline storage).
  char content[IREE_TOKENIZER_DECODER_STRIP_MAX_CONTENT_LENGTH];
  iree_host_size_t content_length;
  // Number of leading occurrences to strip.
  iree_host_size_t start_count;
  // Number of trailing occurrences to strip (not yet supported).
  iree_host_size_t stop_count;
} iree_tokenizer_decoder_strip_t;

typedef struct iree_tokenizer_decoder_strip_state_t {
  iree_tokenizer_decoder_state_t base;
  // Copied from decoder for cache locality.
  char content[IREE_TOKENIZER_DECODER_STRIP_MAX_CONTENT_LENGTH];
  iree_host_size_t content_length;
  // Number of leading occurrences to strip per token.
  iree_host_size_t start_count;
} iree_tokenizer_decoder_strip_state_t;

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_strip_vtable;

iree_status_t iree_tokenizer_decoder_strip_allocate(
    iree_string_view_t content, iree_host_size_t start_count,
    iree_host_size_t stop_count, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;

  if (content.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strip content must not be empty");
  }
  if (content.size > IREE_TOKENIZER_DECODER_STRIP_MAX_CONTENT_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strip content length (%zu) exceeds maximum (%d)",
                            content.size,
                            IREE_TOKENIZER_DECODER_STRIP_MAX_CONTENT_LENGTH);
  }
  if (stop_count > 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Strip decoder with stop_count > 0 not supported (requires buffering)");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_decoder_strip_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  iree_tokenizer_decoder_initialize(
      &decoder->base, &iree_tokenizer_decoder_strip_vtable,
      sizeof(iree_tokenizer_decoder_strip_state_t),
      IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS);
  decoder->allocator = allocator;

  memcpy(decoder->content, content.data, content.size);
  decoder->content_length = content.size;
  decoder->start_count = start_count;
  // stop_count is always 0 here (>0 sets UNIMPLEMENTED status above).
  // Field is retained for API stability; reserved for future implementation.
  IREE_ASSERT(stop_count == 0);
  decoder->stop_count = stop_count;

  *out_decoder = &decoder->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_strip_destroy(
    iree_tokenizer_decoder_t* base_decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_strip_t* decoder =
      (iree_tokenizer_decoder_strip_t*)base_decoder;
  iree_allocator_t allocator = decoder->allocator;
  iree_allocator_free(allocator, decoder);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_strip_state_initialize(
    const iree_tokenizer_decoder_t* base_decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_decoder_strip_t* decoder =
      (const iree_tokenizer_decoder_strip_t*)base_decoder;
  iree_tokenizer_decoder_strip_state_t* state =
      (iree_tokenizer_decoder_strip_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.decoder = base_decoder;

  // Copy config to state for cache-friendly hot path.
  memcpy(state->content, decoder->content, decoder->content_length);
  state->content_length = decoder->content_length;
  state->start_count = decoder->start_count;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_strip_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_strip_state_process(
    iree_tokenizer_decoder_state_t* base_state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  iree_tokenizer_decoder_strip_state_t* state =
      (iree_tokenizer_decoder_strip_state_t*)base_state;

  const char* content = state->content;
  const iree_host_size_t content_length = state->content_length;
  const iree_host_size_t start_count = state->start_count;
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  for (iree_host_size_t i = 0; i < token_strings.count; ++i) {
    iree_string_view_t token = token_strings.values[i];
    iree_host_size_t token_position = 0;

    // Strip up to start_count leading occurrences of content from THIS token.
    // Each token is processed independently (per HuggingFace semantics).
    iree_host_size_t stripped = 0;
    while (stripped < start_count &&
           token_position + content_length <= token.size) {
      if (memcmp(token.data + token_position, content, content_length) == 0) {
        token_position += content_length;
        ++stripped;
      } else {
        // First non-matching content ends stripping for this token.
        break;
      }
    }

    // Copy remaining token to output.
    // SAFETY: token_position <= token.size is guaranteed by the while loop
    // condition above (token_position + content_length <= token.size) and
    // content_length >= 1 (validated at construction time). The subtraction
    // cannot underflow.
    iree_host_size_t remaining = token.size - token_position;
    // SAFETY: bytes_written <= output.size is maintained because we only
    // increment bytes_written by remaining after checking remaining <=
    // available.
    iree_host_size_t available = output.size - bytes_written;
    if (remaining > available) {
      // Buffer full - cannot fit remainder.
      break;
    }

    // Use memmove - source and destination may overlap when used in sequence
    // decoder (same buffer for input and output).
    memmove(output.data + bytes_written, token.data + token_position,
            remaining);
    bytes_written += remaining;
    ++strings_consumed;
  }

  *out_strings_consumed = strings_consumed;
  *out_bytes_written = bytes_written;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_strip_state_finalize(
    iree_tokenizer_decoder_state_t* state, iree_mutable_string_view_t output,
    iree_host_size_t* out_written) {
  // No buffered data to flush for start-only stripping.
  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_strip_state_has_pending(
    const iree_tokenizer_decoder_state_t* state) {
  // Start-only stripping never has pending data.
  return false;
}

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_strip_vtable = {
        .destroy = iree_tokenizer_decoder_strip_destroy,
        .state_initialize = iree_tokenizer_decoder_strip_state_initialize,
        .state_deinitialize = iree_tokenizer_decoder_strip_state_deinitialize,
        .state_process = iree_tokenizer_decoder_strip_state_process,
        .state_finalize = iree_tokenizer_decoder_strip_state_finalize,
        .state_has_pending = iree_tokenizer_decoder_strip_state_has_pending,
        .flags = IREE_TOKENIZER_DECODER_FLAG_SHRINKING,
};
