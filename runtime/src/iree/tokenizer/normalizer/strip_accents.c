// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/strip_accents.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Strip Accents Normalizer Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_normalizer_strip_accents_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
} iree_tokenizer_normalizer_strip_accents_t;

// State for streaming strip_accents processing.
// This normalizer only filters (removes) combining marks, never expands output.
// Since output is always <= input, no pending output buffer is needed.
//
// Per the normalizer interface contract, callers guarantee input arrives on
// codepoint boundaries. We do not buffer incomplete UTF-8 sequences.
typedef struct iree_tokenizer_normalizer_strip_accents_state_t {
  iree_tokenizer_normalizer_state_t base;
  // No additional state needed - this is a stateless filter.
} iree_tokenizer_normalizer_strip_accents_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_strip_accents_vtable;

iree_status_t iree_tokenizer_normalizer_strip_accents_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  iree_tokenizer_normalizer_strip_accents_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_strip_accents_vtable,
      sizeof(iree_tokenizer_normalizer_strip_accents_state_t));
  normalizer->allocator = allocator;

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_strip_accents_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_strip_accents_t* self =
      (iree_tokenizer_normalizer_strip_accents_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_strip_accents_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_strip_accents_state_t* state =
      (iree_tokenizer_normalizer_strip_accents_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_strip_accents_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Returns true if byte is ASCII (high bit clear).
static inline bool iree_tokenizer_strip_accents_is_ascii(uint8_t byte) {
  return (byte & 0x80) == 0;
}

static iree_status_t iree_tokenizer_normalizer_strip_accents_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  (void)flags;  // Strip accents is stateless - just filters marks.

  const uint8_t* in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Main processing loop with ASCII fast path.
  while (in_ptr < in_end && out_ptr < out_end) {
    // ASCII fast path: all ASCII bytes pass through unchanged.
    // Combining marks are always >= U+0300 (non-ASCII), so no ASCII byte
    // is ever a mark that needs filtering.
    while (in_ptr < in_end && out_ptr < out_end &&
           iree_tokenizer_strip_accents_is_ascii(*in_ptr)) {
      *out_ptr++ = *in_ptr++;
    }

    // If we hit end of input or output, we're done with ASCII run.
    if (in_ptr >= in_end || out_ptr >= out_end) {
      break;
    }

    // Non-ASCII byte encountered - decode codepoint and check if it's a mark.
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

    // Check if this is a combining mark.
    if (iree_unicode_is_mark(codepoint)) {
      // Skip the mark - don't copy to output.
      in_ptr += position;
    } else {
      // Not a mark - copy to output.
      // Check if we have enough output space.
      iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
      if (output_available < position) {
        // Output buffer full - stop without consuming this codepoint.
        break;
      }
      memcpy(out_ptr, in_ptr, position);
      out_ptr += position;
      in_ptr += position;
    }
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_strip_accents_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  // No pending data - this normalizer only filters without buffering.
  // Per the normalizer contract, input arrives on codepoint boundaries, so
  // there's no incomplete UTF-8 to handle here.
  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_strip_accents_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  // This normalizer only filters (removes marks), never buffers output.
  // There is never pending data.
  return false;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_strip_accents_vtable = {
        .destroy = iree_tokenizer_normalizer_strip_accents_destroy,
        .state_initialize =
            iree_tokenizer_normalizer_strip_accents_state_initialize,
        .state_deinitialize =
            iree_tokenizer_normalizer_strip_accents_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_strip_accents_state_process,
        .state_finalize =
            iree_tokenizer_normalizer_strip_accents_state_finalize,
        .state_has_pending =
            iree_tokenizer_normalizer_strip_accents_state_has_pending,
};
