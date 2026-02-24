// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/nfd.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// NFD Normalizer Implementation
//===----------------------------------------------------------------------===//

// Maximum combining sequence length (starter + combining marks).
// Unicode Stream-Safe Text Format (UAX #15 §12) limits combining character
// sequences to 30 trailing characters. We use 32 for margin.
// Inputs exceeding this are provably non-conformant Unicode text.
#define IREE_TOKENIZER_NFD_MAX_SEQUENCE 32

typedef struct iree_tokenizer_normalizer_nfd_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
} iree_tokenizer_normalizer_nfd_t;

// State for streaming NFD processing.
//
// Buffers a combining sequence (starter + combining marks) until the next
// starter arrives, then canonically orders and emits it (without composition).
typedef struct iree_tokenizer_normalizer_nfd_state_t {
  iree_tokenizer_normalizer_state_t base;
  // Combining sequence buffer: codepoints awaiting emission.
  // Entry 0 is the starter (CCC=0), entries 1..count-1 are combining marks.
  uint32_t sequence[IREE_TOKENIZER_NFD_MAX_SEQUENCE];
  uint8_t sequence_count;
  // Position within the ordered sequence during emit phase.
  // When emit_position < sequence_count, we're in the middle of emitting.
  uint8_t emit_position;
  // Partial UTF-8 encoding for a codepoint that didn't fully fit in output.
  uint8_t pending_utf8[4];
  uint8_t pending_utf8_count;
  uint8_t pending_utf8_position;
  // Buffered decomposed codepoints from an input codepoint being processed.
  // Needed because Hangul decomposes to multiple starters (CCC=0) and we may
  // not be able to ingest all of them in one call due to output capacity.
  uint32_t decomposed_buffer[4];
  uint8_t decomposed_count;
  uint8_t decomposed_position;
} iree_tokenizer_normalizer_nfd_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_nfd_vtable;

iree_status_t iree_tokenizer_normalizer_nfd_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  iree_tokenizer_normalizer_nfd_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_nfd_vtable,
      sizeof(iree_tokenizer_normalizer_nfd_state_t));
  normalizer->allocator = allocator;

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_nfd_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_nfd_t* self =
      (iree_tokenizer_normalizer_nfd_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_nfd_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_nfd_state_t* state =
      (iree_tokenizer_normalizer_nfd_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_nfd_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Canonical Ordering and Emission Helpers
//===----------------------------------------------------------------------===//

// Applies canonical ordering (stable sort by CCC) to the combining sequence.
// Unlike NFC, we do not compose — just order the combining marks.
static void iree_tokenizer_nfd_order_sequence(
    iree_tokenizer_normalizer_nfd_state_t* state) {
  if (state->sequence_count <= 1) return;

  // Canonical ordering: insertion sort combining marks by CCC.
  // Stable: marks with equal CCC preserve their relative order.
  // Entry 0 is normally the starter (CCC=0). For defective sequences
  // (no preceding starter), entry 0 has CCC > 0 and must participate.
  uint8_t sort_start = (iree_unicode_ccc(state->sequence[0]) > 0) ? 0 : 1;
  for (uint8_t i = sort_start + 1; i < state->sequence_count; ++i) {
    uint32_t codepoint = state->sequence[i];
    uint8_t ccc = iree_unicode_ccc(codepoint);
    uint8_t j = i;
    while (j > sort_start && iree_unicode_ccc(state->sequence[j - 1]) > ccc) {
      state->sequence[j] = state->sequence[j - 1];
      --j;
    }
    state->sequence[j] = codepoint;
  }
}

// Emits pending UTF-8 bytes to the output buffer.
static iree_host_size_t iree_tokenizer_nfd_emit_pending_utf8(
    iree_tokenizer_normalizer_nfd_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  iree_host_size_t available =
      state->pending_utf8_count - state->pending_utf8_position;
  iree_host_size_t to_write =
      available < output_capacity ? available : output_capacity;
  if (to_write > 0) {
    memcpy(output, state->pending_utf8 + state->pending_utf8_position,
           to_write);
    state->pending_utf8_position += (uint8_t)to_write;
    if (state->pending_utf8_position >= state->pending_utf8_count) {
      state->pending_utf8_count = 0;
      state->pending_utf8_position = 0;
    }
  }
  return to_write;
}

// Emits a single codepoint to the output buffer.
static iree_host_size_t iree_tokenizer_nfd_emit_codepoint(
    iree_tokenizer_normalizer_nfd_state_t* state, uint32_t codepoint,
    uint8_t* output, iree_host_size_t output_capacity) {
  int encoded_length = iree_unicode_utf8_encoded_length(codepoint);
  IREE_ASSERT(encoded_length > 0,
              "invalid codepoint U+%04X in NFD emit (internal bug)", codepoint);
  if (encoded_length <= 0) return 0;

  if ((iree_host_size_t)encoded_length <= output_capacity) {
    iree_unicode_utf8_encode(codepoint, (char*)output);
    return (iree_host_size_t)encoded_length;
  }

  // Doesn't fully fit - encode to pending_utf8, write what fits.
  IREE_ASSERT(state->pending_utf8_count == 0 ||
                  state->pending_utf8_position >= state->pending_utf8_count,
              "pending_utf8 overwritten before fully drained");
  iree_unicode_utf8_encode(codepoint, (char*)state->pending_utf8);
  state->pending_utf8_count = (uint8_t)encoded_length;
  state->pending_utf8_position = 0;
  return iree_tokenizer_nfd_emit_pending_utf8(state, output, output_capacity);
}

// Emits codepoints from the ordered sequence starting at emit_position.
static iree_host_size_t iree_tokenizer_nfd_emit_sequence(
    iree_tokenizer_normalizer_nfd_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  uint8_t* out_ptr = output;
  iree_host_size_t remaining = output_capacity;

  while (state->emit_position < state->sequence_count && remaining > 0) {
    uint32_t codepoint = state->sequence[state->emit_position];
    iree_host_size_t written =
        iree_tokenizer_nfd_emit_codepoint(state, codepoint, out_ptr, remaining);
    out_ptr += written;
    remaining -= written;
    ++state->emit_position;

    // If we have pending_utf8 that couldn't be fully written, stop.
    if (state->pending_utf8_count > state->pending_utf8_position) break;
  }

  // If fully emitted, reset the sequence.
  if (state->emit_position >= state->sequence_count &&
      state->pending_utf8_count == 0) {
    state->sequence_count = 0;
    state->emit_position = 0;
  }

  return (iree_host_size_t)(out_ptr - output);
}

// Orders the current sequence and begins emitting.
static iree_host_size_t iree_tokenizer_nfd_flush_and_emit(
    iree_tokenizer_normalizer_nfd_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  if (state->sequence_count == 0) return 0;
  iree_tokenizer_nfd_order_sequence(state);
  state->emit_position = 0;
  return iree_tokenizer_nfd_emit_sequence(state, output, output_capacity);
}

// Drains all pending output.
static iree_host_size_t iree_tokenizer_nfd_drain_pending(
    iree_tokenizer_normalizer_nfd_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  uint8_t* out_ptr = output;
  iree_host_size_t remaining = output_capacity;

  // Drain partial UTF-8 from a previous emit.
  if (state->pending_utf8_count > state->pending_utf8_position) {
    iree_host_size_t written =
        iree_tokenizer_nfd_emit_pending_utf8(state, out_ptr, remaining);
    out_ptr += written;
    remaining -= written;
    if (state->pending_utf8_count > state->pending_utf8_position) {
      return (iree_host_size_t)(out_ptr - output);
    }
  }

  // Reset sequence if all codepoints were already emitted.
  if (state->emit_position > 0 &&
      state->emit_position >= state->sequence_count) {
    state->sequence_count = 0;
    state->emit_position = 0;
  }

  // Continue emitting a partially-emitted sequence.
  if (state->emit_position > 0 &&
      state->emit_position < state->sequence_count) {
    iree_host_size_t written =
        iree_tokenizer_nfd_emit_sequence(state, out_ptr, remaining);
    out_ptr += written;
  }

  return (iree_host_size_t)(out_ptr - output);
}

//===----------------------------------------------------------------------===//
// Codepoint Ingestion
//===----------------------------------------------------------------------===//

// Ingests a single decomposed codepoint into the combining sequence.
//
// For starters (CCC=0): flushes and emits the previous sequence (if any),
// then starts a new sequence.
//
// For combining marks (CCC>0): appends to the current sequence.
static iree_status_t iree_tokenizer_nfd_ingest_codepoint(
    iree_tokenizer_normalizer_nfd_state_t* state, uint32_t codepoint,
    uint8_t** out_ptr, uint8_t* out_end, bool* out_consumed) {
  *out_consumed = false;
  uint8_t ccc = iree_unicode_ccc(codepoint);
  iree_host_size_t remaining = (iree_host_size_t)(out_end - *out_ptr);

  if (ccc == 0) {
    // Starter character.
    if (state->sequence_count == 0) {
      // No pending sequence - just start a new one.
      state->sequence[0] = codepoint;
      state->sequence_count = 1;
      *out_consumed = true;
    } else {
      // Previous sequence pending - flush it first.
      iree_host_size_t written =
          iree_tokenizer_nfd_flush_and_emit(state, *out_ptr, remaining);
      *out_ptr += written;
      if (state->sequence_count > 0) {
        // Couldn't fully flush - can't consume this codepoint yet.
        return iree_ok_status();
      }
      state->sequence[0] = codepoint;
      state->sequence_count = 1;
      *out_consumed = true;
    }
  } else {
    // Combining mark.
    if (state->sequence_count == 0) {
      // Defective combining character sequence (no preceding starter).
      state->sequence[0] = codepoint;
      state->sequence_count = 1;
      *out_consumed = true;
    } else {
      if (state->sequence_count >= IREE_TOKENIZER_NFD_MAX_SEQUENCE) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "NFD combining sequence exceeds Unicode Stream-Safe "
            "limit of %d codepoints",
            IREE_TOKENIZER_NFD_MAX_SEQUENCE);
      }
      state->sequence[state->sequence_count++] = codepoint;
      *out_consumed = true;
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Normalizer Vtable Implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_nfd_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_nfd_state_t* state =
      (iree_tokenizer_normalizer_nfd_state_t*)base_state;

  // When SEGMENT_END is set, we flush pending combining sequence after
  // processing input.
  const bool is_segment_end =
      iree_any_bit_set(flags, IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END);

  const uint8_t* in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Drain any pending output from a previous partial emit.
  iree_host_size_t drained = iree_tokenizer_nfd_drain_pending(
      state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
  out_ptr += drained;
  if (state->pending_utf8_count > state->pending_utf8_position ||
      (state->emit_position > 0 &&
       state->emit_position < state->sequence_count)) {
    // Still have pending output - can't process new input.
    *out_consumed = 0;
    *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
    return iree_ok_status();
  }

  // Main processing loop.
  bool drain_stalled = false;
  while (in_ptr < in_end ||
         state->decomposed_position < state->decomposed_count) {
    // First, drain any buffered decomposed codepoints from a previous call.
    // This is needed for Hangul and other cases where one input codepoint
    // decomposes to multiple output codepoints (potentially multiple starters).
    while (state->decomposed_position < state->decomposed_count) {
      bool consumed = false;
      iree_status_t status = iree_tokenizer_nfd_ingest_codepoint(
          state, state->decomposed_buffer[state->decomposed_position], &out_ptr,
          out_end, &consumed);
      if (!iree_status_is_ok(status)) {
        *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
        *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
        return status;
      }
      if (!consumed) {
        drain_stalled = true;
        break;
      }
      ++state->decomposed_position;
    }
    if (drain_stalled) break;
    // Clear the buffer once fully consumed.
    state->decomposed_count = 0;
    state->decomposed_position = 0;

    // No more input to process.
    if (in_ptr >= in_end) break;

    // ASCII fast path: when no combining sequence is pending, emit ASCII bytes
    // directly. We buffer the last ASCII byte before a non-ASCII byte (or end
    // of input) because ASCII characters can have following combining marks
    // (e.g., 'a' + U+0301 combining acute).
    if (state->sequence_count == 0) {
      while (in_ptr < in_end && out_ptr < out_end && (*in_ptr & 0x80) == 0) {
        if (in_ptr + 1 >= in_end || (*(in_ptr + 1) & 0x80) != 0) {
          // Last ASCII or next is non-ASCII - buffer for potential combining.
          state->sequence[0] = *in_ptr++;
          state->sequence_count = 1;
          break;
        }
        *out_ptr++ = *in_ptr++;
      }
      if (in_ptr >= in_end ||
          (state->sequence_count == 0 && out_ptr >= out_end))
        break;
      if (state->sequence_count > 0 && in_ptr >= in_end) break;
    }

    // Output full - stop processing.
    if (out_ptr >= out_end &&
        (state->sequence_count > 0 ||
         state->pending_utf8_count > state->pending_utf8_position)) {
      break;
    }

    // Decode the next codepoint.
    iree_host_size_t byte_count;
    if ((*in_ptr & 0x80) == 0) {
      byte_count = 1;
    } else {
      byte_count = iree_unicode_utf8_sequence_length(*in_ptr);
    }
    iree_host_size_t available = (iree_host_size_t)(in_end - in_ptr);
    IREE_ASSERT(available >= byte_count,
                "incomplete UTF-8 at end of normalizer input violates "
                "interface contract");

    iree_string_view_t remaining_input = {
        (const char*)in_ptr,
        available,
    };
    iree_host_size_t decode_position = 0;
    uint32_t codepoint =
        iree_unicode_utf8_decode(remaining_input, &decode_position);

    // NFD canonical decomposition: full decomposition to base + combining.
    // Unlike NFC which uses decompose_nfc_canonical (preserves precomposed),
    // NFD uses the full decompose function.
    // Buffer the decomposed codepoints so we can resume if output is full.
    state->decomposed_count =
        (uint8_t)iree_unicode_decompose(codepoint, state->decomposed_buffer);
    state->decomposed_position = 0;

    // Advance input pointer - we've committed to processing this codepoint.
    // The decomposed buffer will be drained on subsequent calls if needed.
    in_ptr += byte_count;
  }

  // At segment end, flush pending combining sequence.
  if (is_segment_end && state->sequence_count > 0 &&
      state->emit_position == 0) {
    iree_host_size_t written = iree_tokenizer_nfd_flush_and_emit(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_nfd_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_nfd_state_t* state =
      (iree_tokenizer_normalizer_nfd_state_t*)base_state;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Drain any partially-emitted output.
  iree_host_size_t drained = iree_tokenizer_nfd_drain_pending(
      state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
  out_ptr += drained;
  if (state->pending_utf8_count > state->pending_utf8_position ||
      (state->emit_position > 0 &&
       state->emit_position < state->sequence_count)) {
    *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
    return iree_ok_status();
  }

  // Drain buffered decomposed codepoints (from Hangul or multi-codepoint
  // decompositions that couldn't be fully ingested in process()).
  while (state->decomposed_position < state->decomposed_count) {
    bool consumed = false;
    iree_status_t status = iree_tokenizer_nfd_ingest_codepoint(
        state, state->decomposed_buffer[state->decomposed_position], &out_ptr,
        out_end, &consumed);
    if (!iree_status_is_ok(status)) {
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return status;
    }
    if (!consumed) {
      // Output full - return what we have.
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
    ++state->decomposed_position;
  }
  state->decomposed_count = 0;
  state->decomposed_position = 0;

  // Order and emit the final combining sequence.
  if (state->sequence_count > 0) {
    iree_host_size_t written = iree_tokenizer_nfd_flush_and_emit(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;
  }

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_nfd_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_nfd_state_t* state =
      (const iree_tokenizer_normalizer_nfd_state_t*)base_state;
  return state->sequence_count > 0 ||
         state->pending_utf8_count > state->pending_utf8_position ||
         state->decomposed_position < state->decomposed_count;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_nfd_vtable = {
        .destroy = iree_tokenizer_normalizer_nfd_destroy,
        .state_initialize = iree_tokenizer_normalizer_nfd_state_initialize,
        .state_deinitialize = iree_tokenizer_normalizer_nfd_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_nfd_state_process,
        .state_finalize = iree_tokenizer_normalizer_nfd_state_finalize,
        .state_has_pending = iree_tokenizer_normalizer_nfd_state_has_pending,
};
