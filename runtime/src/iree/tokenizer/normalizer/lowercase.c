// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/lowercase.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Lowercase Normalizer Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_normalizer_lowercase_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
} iree_tokenizer_normalizer_lowercase_t;

// State for streaming lowercase processing.
// Handles pending output from codepoint expansion (İ → i + combining dot).
//
// Per the normalizer interface contract, callers guarantee input arrives on
// codepoint boundaries. We do not buffer incomplete UTF-8 sequences.
typedef struct iree_tokenizer_normalizer_lowercase_state_t {
  iree_tokenizer_normalizer_state_t base;
  // Pending output bytes from a codepoint that didn't fit in output buffer.
  // Max 8 bytes: İ expands to 2 codepoints, each up to 4 bytes UTF-8.
  uint8_t pending_output[8];
  uint8_t pending_output_count;
  uint8_t pending_output_position;
} iree_tokenizer_normalizer_lowercase_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_lowercase_vtable;

iree_status_t iree_tokenizer_normalizer_lowercase_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  iree_tokenizer_normalizer_lowercase_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_lowercase_vtable,
      sizeof(iree_tokenizer_normalizer_lowercase_state_t));
  normalizer->allocator = allocator;

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_lowercase_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_lowercase_t* self =
      (iree_tokenizer_normalizer_lowercase_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_lowercase_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_lowercase_state_t* state =
      (iree_tokenizer_normalizer_lowercase_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_lowercase_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Returns true if byte is ASCII (high bit clear).
static inline bool iree_tokenizer_is_ascii(uint8_t byte) {
  return (byte & 0x80) == 0;
}

// Lowercases an ASCII byte in-place if it's an uppercase letter.
static inline uint8_t iree_tokenizer_ascii_tolower(uint8_t byte) {
  if (byte >= 'A' && byte <= 'Z') {
    return byte | 0x20;
  }
  return byte;
}

// Emits pending output bytes to the output buffer.
// Returns number of bytes written.
static iree_host_size_t iree_tokenizer_lowercase_emit_pending(
    iree_tokenizer_normalizer_lowercase_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  iree_host_size_t available =
      state->pending_output_count - state->pending_output_position;
  iree_host_size_t to_write =
      available < output_capacity ? available : output_capacity;
  if (to_write > 0) {
    memcpy(output, state->pending_output + state->pending_output_position,
           to_write);
    state->pending_output_position += (uint8_t)to_write;
    if (state->pending_output_position >= state->pending_output_count) {
      state->pending_output_count = 0;
      state->pending_output_position = 0;
    }
  }
  return to_write;
}

// Lowercases a single codepoint and writes the result to pending_output.
// Returns the number of bytes written to pending_output.
static iree_host_size_t iree_tokenizer_lowercase_codepoint(
    iree_tokenizer_normalizer_lowercase_state_t* state, uint32_t codepoint) {
  uint32_t lower[2];
  iree_host_size_t count = iree_unicode_to_lower(codepoint, lower);

  iree_host_size_t total_bytes = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    int encoded = iree_unicode_utf8_encode(
        lower[i], (char*)(state->pending_output + total_bytes));
    if (encoded > 0) {
      total_bytes += (iree_host_size_t)encoded;
    }
  }
  state->pending_output_count = (uint8_t)total_bytes;
  state->pending_output_position = 0;
  return total_bytes;
}

static iree_status_t iree_tokenizer_normalizer_lowercase_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  (void)flags;  // Lowercase is stateless within codepoints, no reset needed.

  iree_tokenizer_normalizer_lowercase_state_t* state =
      (iree_tokenizer_normalizer_lowercase_state_t*)base_state;

  const uint8_t* in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // First, emit any pending output from previous call.
  if (state->pending_output_count > state->pending_output_position) {
    iree_host_size_t written = iree_tokenizer_lowercase_emit_pending(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;
    if (state->pending_output_count > state->pending_output_position) {
      // Output buffer full, couldn't emit all pending.
      *out_consumed = 0;
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Main processing loop with ASCII fast path.
  while (in_ptr < in_end && out_ptr < out_end) {
    // ASCII fast path: process run of ASCII bytes directly without UTF-8
    // decode/encode overhead.
    while (in_ptr < in_end && out_ptr < out_end &&
           iree_tokenizer_is_ascii(*in_ptr)) {
      *out_ptr++ = iree_tokenizer_ascii_tolower(*in_ptr++);
    }

    // If we hit end of input or output, we're done with ASCII run.
    if (in_ptr >= in_end || out_ptr >= out_end) {
      break;
    }

    // Non-ASCII byte encountered - use full Unicode path.
    uint8_t lead_byte = *in_ptr;
    iree_host_size_t seq_length = iree_unicode_utf8_sequence_length(lead_byte);
    iree_host_size_t available = (iree_host_size_t)(in_end - in_ptr);

    // Per normalizer contract, input must be on codepoint boundaries.
    IREE_ASSERT(available >= seq_length,
                "incomplete UTF-8 at end of normalizer input violates "
                "interface contract");

    // Decode the codepoint.
    iree_string_view_t remaining = {
        (const char*)in_ptr,
        (iree_host_size_t)(in_end - in_ptr),
    };
    iree_host_size_t position = 0;
    uint32_t codepoint = iree_unicode_utf8_decode(remaining, &position);
    in_ptr += position;

    // Lowercase and try to emit.
    iree_tokenizer_lowercase_codepoint(state, codepoint);
    iree_host_size_t written = iree_tokenizer_lowercase_emit_pending(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;

    if (state->pending_output_count > state->pending_output_position) {
      // Output buffer full - stop processing.
      break;
    }
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_lowercase_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_lowercase_state_t* state =
      (iree_tokenizer_normalizer_lowercase_state_t*)base_state;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // First, emit any pending output.
  if (state->pending_output_count > state->pending_output_position) {
    iree_host_size_t written = iree_tokenizer_lowercase_emit_pending(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;
    if (state->pending_output_count > state->pending_output_position) {
      // Output buffer full.
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Per the normalizer contract, input arrives on codepoint boundaries, so
  // there's no incomplete UTF-8 to handle here.

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_lowercase_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_lowercase_state_t* state =
      (const iree_tokenizer_normalizer_lowercase_state_t*)base_state;
  // Only pending if we have output that didn't fit in the output buffer.
  // Per the normalizer contract, input arrives on codepoint boundaries, so
  // there's no incomplete UTF-8 to track.
  return state->pending_output_count > state->pending_output_position;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_lowercase_vtable = {
        .destroy = iree_tokenizer_normalizer_lowercase_destroy,
        .state_initialize =
            iree_tokenizer_normalizer_lowercase_state_initialize,
        .state_deinitialize =
            iree_tokenizer_normalizer_lowercase_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_lowercase_state_process,
        .state_finalize = iree_tokenizer_normalizer_lowercase_state_finalize,
        .state_has_pending =
            iree_tokenizer_normalizer_lowercase_state_has_pending,
};
