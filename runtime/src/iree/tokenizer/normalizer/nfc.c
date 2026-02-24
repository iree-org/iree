// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/nfc.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// NFC Normalizer Implementation
//===----------------------------------------------------------------------===//

// Maximum combining sequence length (starter + combining marks).
// Unicode Stream-Safe Text Format (UAX #15 §12) limits combining character
// sequences to 30 trailing characters. We use 32 (matching
// IREE_UNICODE_MAX_COMBINING_SEQUENCE in unicode.c) for margin.
// Inputs exceeding this are provably non-conformant Unicode text.
#define IREE_TOKENIZER_NFC_MAX_SEQUENCE 32

typedef struct iree_tokenizer_normalizer_nfc_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
} iree_tokenizer_normalizer_nfc_t;

// State for streaming NFC processing.
//
// Buffers a combining sequence (starter + combining marks) until the next
// starter arrives, then canonically orders and composes it before emitting.
//
// Per the normalizer interface contract, callers guarantee input arrives on
// codepoint boundaries. We do not buffer incomplete UTF-8 sequences.
typedef struct iree_tokenizer_normalizer_nfc_state_t {
  iree_tokenizer_normalizer_state_t base;
  // Combining sequence buffer: codepoints awaiting composition.
  // Entry 0 is the starter (CCC=0), entries 1..count-1 are combining marks.
  uint32_t sequence[IREE_TOKENIZER_NFC_MAX_SEQUENCE];
  uint8_t sequence_count;
  // Position within the composed sequence during emit phase.
  // When emit_position < sequence_count, we're in the middle of emitting.
  uint8_t emit_position;
  // Partial UTF-8 encoding for a codepoint that didn't fully fit in output.
  // Bounded at 4 bytes (maximum UTF-8 encoding of a single codepoint).
  uint8_t pending_utf8[4];
  uint8_t pending_utf8_count;
  uint8_t pending_utf8_position;
} iree_tokenizer_normalizer_nfc_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_nfc_vtable;

iree_status_t iree_tokenizer_normalizer_nfc_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  iree_tokenizer_normalizer_nfc_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_nfc_vtable,
      sizeof(iree_tokenizer_normalizer_nfc_state_t));
  normalizer->allocator = allocator;

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_nfc_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_nfc_t* self =
      (iree_tokenizer_normalizer_nfc_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_nfc_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_nfc_state_t* state =
      (iree_tokenizer_normalizer_nfc_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_nfc_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Composition and Emission Helpers
//===----------------------------------------------------------------------===//

// Applies canonical ordering (stable sort by CCC) and canonical composition
// to the combining sequence in |state|. After this call, the sequence contains
// the composed result ready for emission.
static void iree_tokenizer_nfc_compose_sequence(
    iree_tokenizer_normalizer_nfc_state_t* state) {
  if (state->sequence_count <= 1) return;

  // Step 1: Canonical ordering — insertion sort combining marks by CCC.
  // Stable: marks with equal CCC preserve their relative order.
  // Normally entry 0 is the starter (CCC=0) and stays fixed, so we sort
  // entries 1..count-1. For defective combining sequences (no preceding
  // starter), entry 0 has CCC > 0 and must participate in the sort.
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

  // Step 2: Canonical composition — greedily compose combining marks with
  // the starter. A mark can compose with the starter if it is not "blocked"
  // (no intervening mark has CCC >= this mark's CCC).
  uint8_t last_ccc = 0;
  uint8_t write_position = 1;

  for (uint8_t i = 1; i < state->sequence_count; ++i) {
    uint32_t codepoint = state->sequence[i];
    uint8_t ccc = iree_unicode_ccc(codepoint);

    bool blocked = (last_ccc >= ccc);
    uint32_t composed = 0;
    if (!blocked) {
      composed = iree_unicode_compose_pair(state->sequence[0], codepoint);
    }

    if (composed != 0) {
      state->sequence[0] = composed;
    } else {
      state->sequence[write_position++] = codepoint;
      last_ccc = ccc;
    }
  }

  state->sequence_count = write_position;
}

// Emits pending UTF-8 bytes to the output buffer.
// Returns the number of bytes written.
static iree_host_size_t iree_tokenizer_nfc_emit_pending_utf8(
    iree_tokenizer_normalizer_nfc_state_t* state, uint8_t* output,
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

// Emits a single codepoint to the output buffer. If it doesn't fully fit,
// stores the remainder in pending_utf8.
// Returns the number of bytes written to output.
static iree_host_size_t iree_tokenizer_nfc_emit_codepoint(
    iree_tokenizer_normalizer_nfc_state_t* state, uint32_t codepoint,
    uint8_t* output, iree_host_size_t output_capacity) {
  int encoded_length = iree_unicode_utf8_encoded_length(codepoint);
  IREE_ASSERT(encoded_length > 0,
              "invalid codepoint U+%04X in NFC emit (internal bug)", codepoint);
  if (encoded_length <= 0) return 0;

  if ((iree_host_size_t)encoded_length <= output_capacity) {
    iree_unicode_utf8_encode(codepoint, (char*)output);
    return (iree_host_size_t)encoded_length;
  }

  // Doesn't fully fit — encode to pending_utf8, write what fits.
  IREE_ASSERT(state->pending_utf8_count == 0 ||
                  state->pending_utf8_position >= state->pending_utf8_count,
              "pending_utf8 overwritten before fully drained");
  iree_unicode_utf8_encode(codepoint, (char*)state->pending_utf8);
  state->pending_utf8_count = (uint8_t)encoded_length;
  state->pending_utf8_position = 0;
  return iree_tokenizer_nfc_emit_pending_utf8(state, output, output_capacity);
}

// Emits codepoints from the composed sequence starting at emit_position.
// Returns the number of bytes written. Updates emit_position.
static iree_host_size_t iree_tokenizer_nfc_emit_sequence(
    iree_tokenizer_normalizer_nfc_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  uint8_t* out_ptr = output;
  iree_host_size_t remaining = output_capacity;

  while (state->emit_position < state->sequence_count && remaining > 0) {
    uint32_t codepoint = state->sequence[state->emit_position];
    iree_host_size_t written =
        iree_tokenizer_nfc_emit_codepoint(state, codepoint, out_ptr, remaining);
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

// Composes the current sequence and begins emitting.
// Returns the number of bytes written.
static iree_host_size_t iree_tokenizer_nfc_flush_and_emit(
    iree_tokenizer_normalizer_nfc_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  if (state->sequence_count == 0) return 0;
  iree_tokenizer_nfc_compose_sequence(state);
  state->emit_position = 0;
  return iree_tokenizer_nfc_emit_sequence(state, output, output_capacity);
}

// Drains all pending output (pending_utf8 + partially-emitted sequence).
// Returns the number of bytes written.
static iree_host_size_t iree_tokenizer_nfc_drain_pending(
    iree_tokenizer_normalizer_nfc_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  uint8_t* out_ptr = output;
  iree_host_size_t remaining = output_capacity;

  // Drain partial UTF-8 from a previous emit.
  if (state->pending_utf8_count > state->pending_utf8_position) {
    iree_host_size_t written =
        iree_tokenizer_nfc_emit_pending_utf8(state, out_ptr, remaining);
    out_ptr += written;
    remaining -= written;
    if (state->pending_utf8_count > state->pending_utf8_position) {
      return (iree_host_size_t)(out_ptr - output);
    }
  }

  // After draining pending UTF-8, check if all codepoints in the sequence were
  // already emitted (emit_position reached sequence_count). If so, the sequence
  // is fully written and can be reset. This handles the case where
  // emit_sequence advanced past all codepoints but couldn't reset because
  // pending_utf8 was still non-empty at that time.
  if (state->emit_position > 0 &&
      state->emit_position >= state->sequence_count) {
    state->sequence_count = 0;
    state->emit_position = 0;
  }

  // Continue emitting a partially-emitted composed sequence.
  if (state->emit_position > 0 &&
      state->emit_position < state->sequence_count) {
    iree_host_size_t written =
        iree_tokenizer_nfc_emit_sequence(state, out_ptr, remaining);
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
// then starts a new sequence. May also compose directly with the previous
// single starter (Hangul Jamo composition).
//
// For combining marks (CCC>0): appends to the current sequence.
//
// Sets |*out_consumed| to true if the codepoint was ingested (state updated).
// If false, the caller should not advance the input pointer — output needs
// draining first.
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the combining sequence overflows.
static iree_status_t iree_tokenizer_nfc_ingest_codepoint(
    iree_tokenizer_normalizer_nfc_state_t* state, uint32_t codepoint,
    uint8_t** out_ptr, uint8_t* out_end, bool* out_consumed) {
  *out_consumed = false;
  uint8_t ccc = iree_unicode_ccc(codepoint);
  iree_host_size_t remaining = (iree_host_size_t)(out_end - *out_ptr);

  if (ccc == 0) {
    // Starter character.
    if (state->sequence_count == 0) {
      // No pending sequence — just start a new one.
      state->sequence[0] = codepoint;
      state->sequence_count = 1;
      *out_consumed = true;
    } else if (state->sequence_count == 1) {
      // Single starter pending — try direct composition (handles Hangul L+V,
      // LV+T, and rare canonical pairs between starters).
      uint32_t composed =
          iree_unicode_compose_pair(state->sequence[0], codepoint);
      if (composed != 0) {
        state->sequence[0] = composed;
        *out_consumed = true;
      } else {
        // Can't compose — emit the old starter, start new sequence.
        iree_host_size_t written = iree_tokenizer_nfc_emit_codepoint(
            state, state->sequence[0], *out_ptr, remaining);
        if (written == 0 && remaining == 0) {
          // No output space — can't make progress.
          return iree_ok_status();
        }
        *out_ptr += written;
        state->sequence[0] = codepoint;
        state->sequence_count = 1;
        *out_consumed = true;
      }
    } else {
      // Starter + combining marks pending — the new starter is blocked by
      // the marks, so flush the full sequence first.
      iree_host_size_t written =
          iree_tokenizer_nfc_flush_and_emit(state, *out_ptr, remaining);
      *out_ptr += written;
      if (state->sequence_count > 0) {
        // Couldn't fully flush — can't consume this codepoint yet.
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
      if (state->sequence_count >= IREE_TOKENIZER_NFC_MAX_SEQUENCE) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "NFC combining sequence exceeds Unicode Stream-Safe "
            "limit of %d codepoints",
            IREE_TOKENIZER_NFC_MAX_SEQUENCE);
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

static iree_status_t iree_tokenizer_normalizer_nfc_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_nfc_state_t* state =
      (iree_tokenizer_normalizer_nfc_state_t*)base_state;

  // When SEGMENT_END is set, we flush the pending combining sequence after
  // processing input (the combining marks are "orphaned" with no following
  // character to compose with).
  const bool is_segment_end =
      iree_any_bit_set(flags, IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END);

  const uint8_t* in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Drain any pending output from a previous partial emit.
  iree_host_size_t drained = iree_tokenizer_nfc_drain_pending(
      state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
  out_ptr += drained;
  if (state->pending_utf8_count > state->pending_utf8_position ||
      (state->emit_position > 0 &&
       state->emit_position < state->sequence_count)) {
    // Still have pending output — can't process new input.
    *out_consumed = 0;
    *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
    return iree_ok_status();
  }

  // Main processing loop.
  while (in_ptr < in_end) {
    // ASCII fast path: when no combining sequence is pending, emit ASCII bytes
    // directly. We buffer the last ASCII byte before a non-ASCII byte (or end
    // of input) because ASCII characters can compose with following combining
    // marks (e.g., 'a' + U+0301 → U+00E1 'á').
    if (state->sequence_count == 0) {
      while (in_ptr < in_end && out_ptr < out_end && (*in_ptr & 0x80) == 0) {
        if (in_ptr + 1 >= in_end || (*(in_ptr + 1) & 0x80) != 0) {
          // Last ASCII or next is non-ASCII — buffer for potential composition.
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

    // Output full — stop processing. When sequence_count > 0, any
    // non-composable input would need to emit the buffered sequence first,
    // which requires output space. We also stop if pending_utf8 is occupied to
    // prevent clobbering.
    if (out_ptr >= out_end &&
        (state->sequence_count > 0 ||
         state->pending_utf8_count > state->pending_utf8_position)) {
      break;
    }

    // Decode the next codepoint without advancing in_ptr yet.
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

    // NFC canonical decomposition: Hangul, CJK compatibility, NFC_QC=No.
    uint32_t decomposed[4];
    iree_host_size_t decomposed_count =
        iree_unicode_decompose_nfc_canonical(codepoint, decomposed);

    // Ingest all decomposed codepoints.
    bool all_consumed = true;
    for (iree_host_size_t d = 0; d < decomposed_count; ++d) {
      bool consumed = false;
      iree_status_t status = iree_tokenizer_nfc_ingest_codepoint(
          state, decomposed[d], &out_ptr, out_end, &consumed);
      if (!iree_status_is_ok(status)) {
        *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
        *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
        return status;
      }
      if (!consumed) {
        all_consumed = false;
        break;
      }
    }

    if (!all_consumed) break;
    in_ptr += byte_count;
  }

  // At segment end, flush pending combining sequence. The combining marks are
  // orphaned (no following character to compose with) so we emit as-is.
  if (is_segment_end && state->sequence_count > 0 &&
      state->emit_position == 0) {
    iree_host_size_t written = iree_tokenizer_nfc_flush_and_emit(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_nfc_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_nfc_state_t* state =
      (iree_tokenizer_normalizer_nfc_state_t*)base_state;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Drain any partially-emitted output.
  iree_host_size_t drained = iree_tokenizer_nfc_drain_pending(
      state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
  out_ptr += drained;
  if (state->pending_utf8_count > state->pending_utf8_position ||
      (state->emit_position > 0 &&
       state->emit_position < state->sequence_count)) {
    *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
    return iree_ok_status();
  }

  // Compose and emit the final combining sequence.
  if (state->sequence_count > 0) {
    iree_host_size_t written = iree_tokenizer_nfc_flush_and_emit(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;
  }

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_nfc_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_nfc_state_t* state =
      (const iree_tokenizer_normalizer_nfc_state_t*)base_state;
  return state->sequence_count > 0 ||
         state->pending_utf8_count > state->pending_utf8_position;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_nfc_vtable = {
        .destroy = iree_tokenizer_normalizer_nfc_destroy,
        .state_initialize = iree_tokenizer_normalizer_nfc_state_initialize,
        .state_deinitialize = iree_tokenizer_normalizer_nfc_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_nfc_state_process,
        .state_finalize = iree_tokenizer_normalizer_nfc_state_finalize,
        .state_has_pending = iree_tokenizer_normalizer_nfc_state_has_pending,
};
