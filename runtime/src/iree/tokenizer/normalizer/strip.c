// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/strip.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Strip Normalizer Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_normalizer_strip_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
  bool strip_left;
  bool strip_right;
} iree_tokenizer_normalizer_strip_t;

// Size of the literal buffer for pending whitespace.
// Mixed whitespace (e.g., "\r\n", " \t") is stored literally.
// When this fills with homogeneous content, we switch to RLE mode.
// 256 bytes handles most real-world cases; RLE overflow handles very long
// runs of the same whitespace character.
#define IREE_TOKENIZER_STRIP_PENDING_BUFFER_SIZE 256

// State for streaming strip processing.
// Uses hybrid buffering: literal buffer for mixed whitespace, RLE overflow
// for long homogeneous runs. This handles both mixed trailing whitespace
// (e.g., "\r\n") and very long runs (e.g., 300 spaces) efficiently.
//
// When non-whitespace follows pending whitespace, we know it's intermediate and
// emit it. On finalize/segment_end, pending whitespace is discarded (trailing).
//
// Per the normalizer interface contract, callers guarantee input arrives on
// codepoint boundaries. We do not buffer incomplete UTF-8 sequences.
typedef struct iree_tokenizer_normalizer_strip_state_t {
  iree_tokenizer_normalizer_state_t base;

  // True if we're still at the start (haven't emitted any non-whitespace).
  bool at_start;

  // For strip_right: pending whitespace that might be trailing.
  // Hybrid approach: literal buffer fills first, then RLE for overflow.
  struct {
    // Literal buffer for mixed whitespace (normal case).
    uint8_t buffer[IREE_TOKENIZER_STRIP_PENDING_BUFFER_SIZE];
    uint32_t length;  // Bytes used in buffer.
    // Bytes emitted from buffer (for draining).
    uint32_t emitted;

    // RLE overflow for homogeneous whitespace that exceeds buffer.
    // Only used when buffer is full and all codepoints are the same.
    // Additional repetitions beyond buffer (0 if not in RLE mode).
    uint32_t rle_count;
    // RLE repetitions emitted (for draining).
    uint32_t rle_emitted;
    // The UTF-8 sequence being repeated.
    uint8_t rle_bytes[4];
    uint8_t rle_length;  // Length of rle_bytes (1-4).

    // True if we've started draining (intermediate WS confirmed).
    bool draining;
  } pending;
} iree_tokenizer_normalizer_strip_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_strip_vtable;

iree_status_t iree_tokenizer_normalizer_strip_allocate(
    bool strip_left, bool strip_right, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  iree_tokenizer_normalizer_strip_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_strip_vtable,
      sizeof(iree_tokenizer_normalizer_strip_state_t));
  normalizer->allocator = allocator;
  normalizer->strip_left = strip_left;
  normalizer->strip_right = strip_right;

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_strip_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_strip_t* self =
      (iree_tokenizer_normalizer_strip_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_strip_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_strip_state_t* state =
      (iree_tokenizer_normalizer_strip_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  state->at_start = true;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_strip_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Returns true if byte is ASCII (high bit clear).
static inline bool iree_tokenizer_strip_is_ascii(uint8_t byte) {
  return (byte & 0x80) == 0;
}

// Returns true if ASCII byte is whitespace.
static inline bool iree_tokenizer_strip_is_ascii_whitespace(uint8_t byte) {
  return byte == ' ' || (byte >= 0x09 && byte <= 0x0D);
}

// Clears pending whitespace state.
static void iree_tokenizer_normalizer_strip_clear_pending(
    iree_tokenizer_normalizer_strip_state_t* state) {
  state->pending.length = 0;
  state->pending.emitted = 0;
  state->pending.rle_count = 0;
  state->pending.rle_emitted = 0;
  state->pending.rle_length = 0;
  state->pending.draining = false;
}

// Returns true if the pending buffer contains only repetitions of the given
// byte sequence. For example, if bytes="\xE3\x80\x80" (U+3000) and length=3,
// returns true if buffer contains only that 3-byte sequence repeated.
static bool iree_tokenizer_normalizer_strip_buffer_is_homogeneous(
    const iree_tokenizer_normalizer_strip_state_t* state, const uint8_t* bytes,
    iree_host_size_t length) {
  if (state->pending.length % length != 0) return false;
  for (uint32_t i = 0; i < state->pending.length; i += (uint32_t)length) {
    if (memcmp(state->pending.buffer + i, bytes, length) != 0) return false;
  }
  return true;
}

// Adds whitespace bytes to pending buffer.
// Returns true on success, false if buffer overflow (shouldn't happen with
// reasonable limits, but we fail loudly rather than silently lose data).
static bool iree_tokenizer_normalizer_strip_add_pending(
    iree_tokenizer_normalizer_strip_state_t* state, const uint8_t* bytes,
    iree_host_size_t length) {
  // If we're already in RLE mode, we can only extend if bytes match.
  if (state->pending.rle_count > 0) {
    // RLE mode: can only add sequences that match the RLE pattern.
    if (length == state->pending.rle_length &&
        memcmp(bytes, state->pending.rle_bytes, length) == 0) {
      state->pending.rle_count++;
      return true;
    }
    // Different sequence in RLE mode - can't extend. This is a limit.
    // In practice, very long mixed whitespace is rare.
    return false;
  }

  // Normal mode: try to add to buffer.
  if (state->pending.length + length <=
      IREE_TOKENIZER_STRIP_PENDING_BUFFER_SIZE) {
    memcpy(state->pending.buffer + state->pending.length, bytes, length);
    state->pending.length += (uint32_t)length;
    return true;
  }

  // Buffer would overflow. Try to switch to RLE mode if homogeneous.
  if (length <= 4 &&
      state->pending.length == IREE_TOKENIZER_STRIP_PENDING_BUFFER_SIZE &&
      iree_tokenizer_normalizer_strip_buffer_is_homogeneous(state, bytes,
                                                            length)) {
    // Switch to RLE mode.
    memcpy(state->pending.rle_bytes, bytes, length);
    state->pending.rle_length = (uint8_t)length;
    state->pending.rle_count = 1;
    return true;
  }

  // Can't extend - mixed whitespace exceeded buffer. This is a hard limit.
  return false;
}

// Emits pending whitespace to output buffer.
// Returns the number of bytes written. Updates emitted counters.
// Call repeatedly until fully drained.
static iree_host_size_t iree_tokenizer_normalizer_strip_emit_pending(
    iree_tokenizer_normalizer_strip_state_t* state, uint8_t* out_ptr,
    uint8_t* out_end) {
  uint8_t* start = out_ptr;

  // First, emit from literal buffer.
  while (state->pending.emitted < state->pending.length) {
    if (out_ptr >= out_end) break;  // Output full.
    *out_ptr++ = state->pending.buffer[state->pending.emitted++];
  }

  // Then, emit RLE overflow (if any).
  while (state->pending.rle_emitted < state->pending.rle_count) {
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    if (output_available < state->pending.rle_length) break;  // Output full.
    memcpy(out_ptr, state->pending.rle_bytes, state->pending.rle_length);
    out_ptr += state->pending.rle_length;
    state->pending.rle_emitted++;
  }

  return (iree_host_size_t)(out_ptr - start);
}

// Returns true if there's pending data not yet fully emitted.
static bool iree_tokenizer_normalizer_strip_has_pending_to_emit(
    const iree_tokenizer_normalizer_strip_state_t* state) {
  return state->pending.emitted < state->pending.length ||
         state->pending.rle_emitted < state->pending.rle_count;
}

static iree_status_t iree_tokenizer_normalizer_strip_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_strip_state_t* state =
      (iree_tokenizer_normalizer_strip_state_t*)base_state;
  const iree_tokenizer_normalizer_strip_t* normalizer =
      (const iree_tokenizer_normalizer_strip_t*)base_state->normalizer;

  const bool is_segment_end =
      iree_any_bit_set(flags, IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END);

  const uint8_t* in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Phase 1: Continue draining pending whitespace if we were mid-drain.
  // We only drain when draining=true, which is set when we've confirmed
  // the pending whitespace is intermediate (non-whitespace follows).
  if (state->pending.draining) {
    out_ptr +=
        iree_tokenizer_normalizer_strip_emit_pending(state, out_ptr, out_end);
    if (iree_tokenizer_normalizer_strip_has_pending_to_emit(state)) {
      // Still draining, output full.
      *out_consumed = 0;
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
    // Finished draining, clear pending.
    iree_tokenizer_normalizer_strip_clear_pending(state);
  }

  // Phase 2: Process input bytes.
  // Key change from lazy consumption: we ALWAYS consume all input.
  // Whitespace is buffered in pending rather than left unconsumed.
  while (in_ptr < in_end) {
    uint8_t byte = *in_ptr;
    iree_host_size_t seq_length = 1;
    bool is_ws = false;

    if (iree_tokenizer_strip_is_ascii(byte)) {
      is_ws = iree_tokenizer_strip_is_ascii_whitespace(byte);
    } else {
      // UTF-8 multi-byte sequence.
      seq_length = iree_unicode_utf8_sequence_length(byte);
      iree_host_size_t available = (iree_host_size_t)(in_end - in_ptr);
      IREE_ASSERT(available >= seq_length,
                  "incomplete UTF-8 at end of normalizer input violates "
                  "interface contract");
      iree_string_view_t remaining =
          iree_make_string_view((const char*)in_ptr, available);
      iree_host_size_t position = 0;
      uint32_t codepoint = iree_unicode_utf8_decode(remaining, &position);
      seq_length = position;
      is_ws = iree_unicode_is_whitespace(codepoint);
    }

    if (is_ws) {
      // Whitespace handling.
      if (state->at_start && normalizer->strip_left) {
        // Leading whitespace - skip entirely.
        in_ptr += seq_length;
        continue;
      }

      if (normalizer->strip_right) {
        // Potential trailing whitespace - buffer it.
        if (!iree_tokenizer_normalizer_strip_add_pending(state, in_ptr,
                                                         seq_length)) {
          // Buffer overflow with mixed whitespace - fail loudly.
          // This is extremely rare in practice (64+ bytes of mixed trailing
          // WS).
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "strip normalizer pending buffer overflow: mixed whitespace "
              "sequence exceeds %" PRIu32 " bytes",
              (uint32_t)IREE_TOKENIZER_STRIP_PENDING_BUFFER_SIZE);
        }
        in_ptr += seq_length;
        continue;
      }

      // strip_right=false: emit whitespace directly.
      iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
      if (output_available < seq_length) break;  // Output full.
      memcpy(out_ptr, in_ptr, seq_length);
      out_ptr += seq_length;
      in_ptr += seq_length;
      continue;
    }

    // Non-whitespace: first emit any pending whitespace (it's intermediate).
    if (state->pending.length > 0 || state->pending.rle_count > 0) {
      // Mark as draining - we've confirmed pending is intermediate.
      state->pending.draining = true;
      out_ptr +=
          iree_tokenizer_normalizer_strip_emit_pending(state, out_ptr, out_end);
      if (iree_tokenizer_normalizer_strip_has_pending_to_emit(state)) {
        // Output full mid-flush. Stop here, don't consume current byte.
        break;
      }
      iree_tokenizer_normalizer_strip_clear_pending(state);
    }

    // Now emit the non-whitespace.
    state->at_start = false;
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    if (output_available < seq_length) break;  // Output full.
    memcpy(out_ptr, in_ptr, seq_length);
    out_ptr += seq_length;
    in_ptr += seq_length;
  }

  // Phase 3: Handle segment end.
  if (is_segment_end) {
    // Pending whitespace is trailing - discard it.
    iree_tokenizer_normalizer_strip_clear_pending(state);
    state->at_start = true;
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_strip_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_strip_state_t* state =
      (iree_tokenizer_normalizer_strip_state_t*)base_state;
  (void)output;

  // With RLE buffering, any pending whitespace at finalize is trailing.
  // Discard it - don't emit.
  iree_tokenizer_normalizer_strip_clear_pending(state);
  state->at_start = true;

  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_strip_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_strip_state_t* state =
      (const iree_tokenizer_normalizer_strip_state_t*)base_state;
  // We have pending data to emit only when we're actively draining
  // (confirmed intermediate whitespace) and haven't finished yet.
  // Buffered potential-trailing whitespace is NOT pending - it will
  // either be emitted when non-WS follows or discarded at finalize.
  return state->pending.draining &&
         iree_tokenizer_normalizer_strip_has_pending_to_emit(state);
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_strip_vtable = {
        .destroy = iree_tokenizer_normalizer_strip_destroy,
        .state_initialize = iree_tokenizer_normalizer_strip_state_initialize,
        .state_deinitialize =
            iree_tokenizer_normalizer_strip_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_strip_state_process,
        .state_finalize = iree_tokenizer_normalizer_strip_state_finalize,
        .state_has_pending = iree_tokenizer_normalizer_strip_state_has_pending,
};
