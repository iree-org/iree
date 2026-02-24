// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/prepend.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Prepend Normalizer Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_normalizer_prepend_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
  // When true, only prepend if input does not start with the prepend string.
  // Implements HuggingFace Metaspace prepend_scheme="first" semantics.
  bool skip_if_prefix_matches;
  // The string to prepend (stored inline after struct).
  iree_host_size_t prepend_length;
  // Prepend data follows immediately after this struct.
} iree_tokenizer_normalizer_prepend_t;

// Decision states for the conditional prepend.
typedef enum {
  // Haven't seen enough input to decide yet (accumulating prefix bytes).
  IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_UNDECIDED = 0,
  // Will prepend: input doesn't start with the prepend string.
  IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_PREPEND,
  // Skip prepend: input already starts with the prepend string.
  IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_SKIP,
} iree_tokenizer_normalizer_prepend_decision_t;

// State for streaming prepend processing.
typedef struct iree_tokenizer_normalizer_prepend_state_t {
  iree_tokenizer_normalizer_state_t base;

  // Hot fields copied from normalizer at init time for cache locality.
  iree_const_byte_span_t prepend;
  bool skip_if_prefix_matches;

  // Decision state for conditional prepend.
  iree_tokenizer_normalizer_prepend_decision_t decision;

  // Prefix comparison buffer: accumulates input bytes until we have enough
  // to compare against the prepend string (only used when
  // skip_if_prefix_matches is true and decision is UNDECIDED).
  iree_host_size_t prefix_buffered;
  uint8_t prefix_buffer[IREE_TOKENIZER_NORMALIZER_PREPEND_MAX_LENGTH];

  // Number of prepend bytes already emitted (for partial writes when output
  // buffer is smaller than prepend string). Only used when decision is PREPEND.
  iree_host_size_t prepend_emitted;

  // Number of prefix buffer bytes already flushed to output. Used after a
  // decision is made to drain the buffered prefix bytes. Only relevant when
  // skip_if_prefix_matches is true.
  iree_host_size_t prefix_flushed;
} iree_tokenizer_normalizer_prepend_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_prepend_vtable;

// Returns pointer to the prepend string data (immediately after struct).
static inline const uint8_t* iree_tokenizer_normalizer_prepend_data(
    const iree_tokenizer_normalizer_prepend_t* normalizer) {
  return (const uint8_t*)(normalizer + 1);
}

iree_status_t iree_tokenizer_normalizer_prepend_allocate(
    iree_string_view_t prepend_string, bool skip_if_prefix_matches,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  if (prepend_string.size > IREE_TOKENIZER_NORMALIZER_PREPEND_MAX_LENGTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepend string length %" PRIhsz " exceeds maximum %d",
        prepend_string.size, IREE_TOKENIZER_NORMALIZER_PREPEND_MAX_LENGTH);
  }

  // Allocate normalizer struct + inline storage for prepend string.
  iree_host_size_t total_size =
      sizeof(iree_tokenizer_normalizer_prepend_t) + prepend_string.size;
  iree_tokenizer_normalizer_prepend_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_prepend_vtable,
      sizeof(iree_tokenizer_normalizer_prepend_state_t));
  normalizer->allocator = allocator;
  normalizer->skip_if_prefix_matches = skip_if_prefix_matches;
  normalizer->prepend_length = prepend_string.size;

  // Copy prepend string to inline storage.
  if (prepend_string.size > 0) {
    memcpy((uint8_t*)(normalizer + 1), prepend_string.data,
           prepend_string.size);
  }

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_prepend_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_prepend_t* self =
      (iree_tokenizer_normalizer_prepend_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_prepend_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_normalizer_prepend_t* self =
      (const iree_tokenizer_normalizer_prepend_t*)normalizer;
  iree_tokenizer_normalizer_prepend_state_t* state =
      (iree_tokenizer_normalizer_prepend_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;

  // Copy hot fields from normalizer to state for cache locality.
  state->prepend = iree_make_const_byte_span(
      iree_tokenizer_normalizer_prepend_data(self), self->prepend_length);
  state->skip_if_prefix_matches = self->skip_if_prefix_matches;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_prepend_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Processing - Unconditional Mode
//===----------------------------------------------------------------------===//

// Emits prepend bytes from state->prepend_emitted onwards into output.
// Returns the number of bytes written. Updates state->prepend_emitted.
static iree_host_size_t iree_tokenizer_normalizer_prepend_emit_prepend(
    iree_tokenizer_normalizer_prepend_state_t* state, uint8_t* out_ptr,
    uint8_t* out_end) {
  iree_host_size_t remaining =
      state->prepend.data_length - state->prepend_emitted;
  iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
  iree_host_size_t to_write =
      remaining < output_available ? remaining : output_available;
  if (to_write > 0) {
    memcpy(out_ptr, state->prepend.data + state->prepend_emitted, to_write);
    state->prepend_emitted += to_write;
  }
  return to_write;
}

//===----------------------------------------------------------------------===//
// Processing - Main Entry Point
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_prepend_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_prepend_state_t* state =
      (iree_tokenizer_normalizer_prepend_state_t*)base_state;

  const uint8_t* in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Phase 1: Make the prepend decision if not yet decided.
  if (state->decision == IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_UNDECIDED) {
    // When FIRST_CONSUMED is set, position 0 of the original input was
    // consumed by a special token. For prepend_scheme="first" (which sets
    // skip_if_prefix_matches), this text is NOT at the "first" position and
    // should not get the prepend.
    if (state->skip_if_prefix_matches &&
        iree_any_bit_set(flags,
                         IREE_TOKENIZER_NORMALIZER_FLAG_FIRST_CONSUMED)) {
      state->decision = IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_SKIP;
    } else if (input.size == 0) {
      // No input yet - can't decide. Report no progress.
      *out_consumed = 0;
      *out_written = 0;
      return iree_ok_status();
    } else if (!state->skip_if_prefix_matches) {
      // Unconditional mode: always prepend.
      state->decision = IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_PREPEND;
    } else {
      // Conditional mode: accumulate bytes and compare against prepend string.
      iree_host_size_t need =
          state->prepend.data_length - state->prefix_buffered;
      iree_host_size_t available = (iree_host_size_t)(in_end - in_ptr);
      iree_host_size_t to_buffer = need < available ? need : available;

      // Copy input bytes to prefix buffer and check for mismatch.
      bool found_mismatch = false;
      for (iree_host_size_t i = 0; i < to_buffer; ++i) {
        uint8_t input_byte = in_ptr[i];
        state->prefix_buffer[state->prefix_buffered + i] = input_byte;
        if (input_byte != state->prepend.data[state->prefix_buffered + i]) {
          // Mismatch found: prepend is needed.
          state->prefix_buffered += i + 1;
          in_ptr += i + 1;
          state->decision = IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_PREPEND;
          found_mismatch = true;
          break;
        }
      }

      if (!found_mismatch) {
        state->prefix_buffered += to_buffer;
        in_ptr += to_buffer;

        if (state->prefix_buffered >= state->prepend.data_length) {
          // Full match: skip the prepend.
          state->decision = IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_SKIP;
        } else {
          // Need more bytes to decide. We consumed input but produce no output.
          *out_consumed =
              (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
          *out_written = 0;
          return iree_ok_status();
        }
      }
    }
  }

  // Phase 2: Emit the prepend string (if decision is PREPEND and not yet done).
  if (state->decision == IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_PREPEND &&
      state->prepend_emitted < state->prepend.data_length) {
    iree_host_size_t written =
        iree_tokenizer_normalizer_prepend_emit_prepend(state, out_ptr, out_end);
    out_ptr += written;

    // If prepend isn't fully emitted yet, return partial progress.
    if (state->prepend_emitted < state->prepend.data_length) {
      *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Phase 3: Flush any buffered prefix bytes (from the comparison phase).
  if (state->prefix_buffered > 0 &&
      state->prefix_flushed < state->prefix_buffered) {
    iree_host_size_t remaining = state->prefix_buffered - state->prefix_flushed;
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    iree_host_size_t to_write =
        remaining < output_available ? remaining : output_available;
    if (to_write > 0) {
      memcpy(out_ptr, state->prefix_buffer + state->prefix_flushed, to_write);
      out_ptr += to_write;
      state->prefix_flushed += to_write;
    }
    if (state->prefix_flushed < state->prefix_buffered) {
      // Buffer not fully flushed yet.
      *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Phase 4: Passthrough remaining input directly to output.
  iree_host_size_t input_remaining = (iree_host_size_t)(in_end - in_ptr);
  iree_host_size_t output_remaining = (iree_host_size_t)(out_end - out_ptr);
  iree_host_size_t to_copy =
      input_remaining < output_remaining ? input_remaining : output_remaining;

  if (to_copy > 0) {
    memcpy(out_ptr, in_ptr, to_copy);
    in_ptr += to_copy;
    out_ptr += to_copy;
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_prepend_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_prepend_state_t* state =
      (iree_tokenizer_normalizer_prepend_state_t*)base_state;

  *out_written = 0;

  // If still undecided at finalize (short input that matched so far but EOF
  // before reaching prepend_length bytes), the input is a PREFIX of the prepend
  // string. Since it's shorter than the prepend string, it doesn't fully match
  // → we should prepend. Transition to PREPEND and flush.
  if (state->decision == IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_UNDECIDED) {
    if (state->prefix_buffered == 0) {
      // Empty input: no prepend, nothing to emit.
      return iree_ok_status();
    }
    // Partial prefix match → prepend is needed (input is shorter than prepend
    // string, so it can't possibly be the full prepend string).
    state->decision = IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_PREPEND;
  }

  // Emit remaining prepend bytes if not yet complete.
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  if (state->decision == IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_PREPEND &&
      state->prepend_emitted < state->prepend.data_length) {
    iree_host_size_t written =
        iree_tokenizer_normalizer_prepend_emit_prepend(state, out_ptr, out_end);
    out_ptr += written;
    if (state->prepend_emitted < state->prepend.data_length) {
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Flush remaining buffered prefix bytes.
  if (state->prefix_buffered > 0 &&
      state->prefix_flushed < state->prefix_buffered) {
    iree_host_size_t remaining = state->prefix_buffered - state->prefix_flushed;
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    iree_host_size_t to_write =
        remaining < output_available ? remaining : output_available;
    if (to_write > 0) {
      memcpy(out_ptr, state->prefix_buffer + state->prefix_flushed, to_write);
      out_ptr += to_write;
      state->prefix_flushed += to_write;
    }
  }

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_prepend_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_prepend_state_t* state =
      (const iree_tokenizer_normalizer_prepend_state_t*)base_state;

  // Pending if we've decided to prepend but haven't finished emitting.
  if (state->decision == IREE_TOKENIZER_NORMALIZER_PREPEND_DECISION_PREPEND &&
      state->prepend_emitted < state->prepend.data_length) {
    return true;
  }

  // Pending if we have buffered prefix bytes not yet flushed.
  if (state->prefix_buffered > 0 &&
      state->prefix_flushed < state->prefix_buffered) {
    return true;
  }

  return false;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_prepend_vtable = {
        .destroy = iree_tokenizer_normalizer_prepend_destroy,
        .state_initialize = iree_tokenizer_normalizer_prepend_state_initialize,
        .state_deinitialize =
            iree_tokenizer_normalizer_prepend_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_prepend_state_process,
        .state_finalize = iree_tokenizer_normalizer_prepend_state_finalize,
        .state_has_pending =
            iree_tokenizer_normalizer_prepend_state_has_pending,
};
