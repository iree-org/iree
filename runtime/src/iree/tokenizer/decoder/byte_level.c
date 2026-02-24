// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/byte_level.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/byte_level_tables.h"

//===----------------------------------------------------------------------===//
// ByteLevel Decoder Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_decoder_byte_level_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
} iree_tokenizer_decoder_byte_level_t;

typedef struct iree_tokenizer_decoder_byte_level_state_t {
  iree_tokenizer_decoder_state_t base;
  // Resume point in current token (for buffer-full mid-token resume).
  iree_host_size_t resume_position;
  // Mapped bytes forming incomplete UTF-8 sequence.
  uint8_t pending_bytes[IREE_UNICODE_UTF8_MAX_BYTE_LENGTH];
  uint8_t pending_count;
} iree_tokenizer_decoder_byte_level_state_t;

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_byte_level_vtable;

iree_status_t iree_tokenizer_decoder_byte_level_allocate(
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_decoder_byte_level_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  iree_tokenizer_decoder_initialize(
      &decoder->base, &iree_tokenizer_decoder_byte_level_vtable,
      sizeof(iree_tokenizer_decoder_byte_level_state_t),
      IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS);
  decoder->allocator = allocator;

  *out_decoder = &decoder->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_byte_level_destroy(
    iree_tokenizer_decoder_t* base_decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_byte_level_t* decoder =
      (iree_tokenizer_decoder_byte_level_t*)base_decoder;
  iree_allocator_t allocator = decoder->allocator;
  iree_allocator_free(allocator, decoder);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_byte_level_state_initialize(
    const iree_tokenizer_decoder_t* base_decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_byte_level_state_t* state =
      (iree_tokenizer_decoder_byte_level_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.decoder = base_decoder;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_byte_level_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// UTF-8 Validation Helpers
//===----------------------------------------------------------------------===//

// Tries to emit a complete UTF-8 sequence from pending bytes.
// Returns bytes written (0 if incomplete or no space).
// For valid sequences, emits the original bytes.
// For invalid sequences, emits one U+FFFD per invalid byte (matching
// HuggingFace behavior where each byte in an invalid sequence gets its own
// replacement character).
static iree_host_size_t iree_tokenizer_byte_level_try_emit_pending(
    iree_tokenizer_decoder_byte_level_state_t* state,
    iree_mutable_string_view_t output, iree_host_size_t write_position) {
  if (state->pending_count == 0) return 0;

  // Get expected sequence length from lead byte.
  iree_host_size_t expected =
      iree_unicode_utf8_sequence_length(state->pending_bytes[0]);

  if (state->pending_count < expected) {
    // Incomplete sequence - wait for more bytes.
    return 0;
  }

  // Validate the sequence without extracting the codepoint.
  bool is_valid = iree_unicode_utf8_is_valid_sequence(state->pending_bytes,
                                                      state->pending_count);

  if (!is_valid) {
    // Invalid sequence - emit one U+FFFD per byte in the invalid sequence.
    iree_host_size_t replacement_count = state->pending_count;
    iree_host_size_t bytes_to_write = replacement_count * 3;

    // Check if we have space.
    if (write_position + bytes_to_write > output.size) {
      return 0;  // No space, don't clear pending.
    }

    // Write N replacement characters.
    for (iree_host_size_t i = 0; i < replacement_count; ++i) {
      output.data[write_position + i * 3 + 0] = (char)0xEF;
      output.data[write_position + i * 3 + 1] = (char)0xBF;
      output.data[write_position + i * 3 + 2] = (char)0xBD;
    }
    state->pending_count = 0;
    return bytes_to_write;
  }

  // Valid sequence or literal U+FFFD - emit as-is.
  iree_host_size_t bytes_to_write = state->pending_count;

  // Check if we have space.
  if (write_position + bytes_to_write > output.size) {
    return 0;  // No space, don't clear pending.
  }

  memcpy(output.data + write_position, state->pending_bytes, bytes_to_write);
  state->pending_count = 0;
  return bytes_to_write;
}

// Flushes any pending bytes as U+FFFD (called before passthrough and at
// finalize). Returns true if successful, false if output buffer is full.
static bool iree_tokenizer_byte_level_flush_pending(
    iree_tokenizer_decoder_byte_level_state_t* state,
    iree_mutable_string_view_t output, iree_host_size_t* write_position) {
  if (state->pending_count == 0) return true;

  // Any pending bytes at flush time are incomplete -> U+FFFD.
  static const uint8_t kUtf8ReplacementChar[3] = {0xEF, 0xBF, 0xBD};
  if (*write_position + 3 > output.size) return false;

  memcpy(output.data + *write_position, kUtf8ReplacementChar, 3);
  *write_position += 3;
  state->pending_count = 0;
  return true;
}

// Handles a mapped byte (from identity or shifted codepoint).
// Adds to pending buffer and tries to emit complete UTF-8 sequences.
// Returns true if processing can continue, false if buffer is full.
static bool iree_tokenizer_byte_level_handle_mapped_byte(
    iree_tokenizer_decoder_byte_level_state_t* state, uint8_t mapped_byte,
    iree_mutable_string_view_t output, iree_host_size_t* write_position) {
  state->pending_bytes[state->pending_count++] = mapped_byte;

  // Check if we have a complete or invalid sequence.
  iree_host_size_t expected =
      iree_unicode_utf8_sequence_length(state->pending_bytes[0]);

  // Case 1: Incomplete sequence (not enough bytes yet, and buffer not full).
  if (state->pending_count < expected &&
      state->pending_count < IREE_UNICODE_UTF8_MAX_BYTE_LENGTH) {
    return true;  // Wait for more bytes.
  }

  // Case 2: Complete or overflowing - try to emit.
  iree_host_size_t emitted = iree_tokenizer_byte_level_try_emit_pending(
      state, output, *write_position);
  if (emitted > 0) {
    *write_position += emitted;
    return true;
  }

  // Case 3: Couldn't emit (buffer full). Undo the byte and signal buffer full.
  state->pending_count--;
  return false;
}

// Processes a run of ASCII identity bytes (0x21-0x7E) with bulk copy.
// Returns true if processing can continue, false if buffer is completely full.
// Advances |*read_position| and |*write_position| by the number of bytes
// copied. If buffer can only fit part of the run, copies what fits and returns
// true.
static bool iree_tokenizer_byte_level_process_ascii_run(
    iree_string_view_t token, iree_host_size_t* read_position,
    iree_mutable_string_view_t output, iree_host_size_t* write_position) {
  // Check if buffer has any space.
  iree_host_size_t available = output.size - *write_position;
  if (available == 0) {
    return false;  // Buffer completely full.
  }

  // Scan for consecutive ASCII identity bytes.
  iree_host_size_t run_length = 1;
  while (*read_position + run_length < token.size && run_length < available) {
    uint8_t byte = (uint8_t)token.data[*read_position + run_length];
    if (byte < 0x21 || byte > 0x7E) break;
    ++run_length;
  }

  // Truncate to available space if needed.
  if (run_length > available) {
    run_length = available;
  }

  // Bulk copy the ASCII run (partial or full).
  memcpy(output.data + *write_position, token.data + *read_position,
         run_length);
  *write_position += run_length;
  *read_position += run_length;
  return true;
}

// Processes a single token, writing decoded output.
// Returns true if token was fully consumed, false if buffer full mid-token.
// On false return, |*read_position| contains the resume position.
static bool iree_tokenizer_byte_level_process_token(
    iree_tokenizer_decoder_byte_level_state_t* state, iree_string_view_t token,
    iree_host_size_t* read_position, iree_mutable_string_view_t output,
    iree_host_size_t* write_position) {
  while (*read_position < token.size) {
    uint8_t first_byte = (uint8_t)token.data[*read_position];

    // ASCII identity fast-path: no pending state and printable ASCII range.
    // This bypasses UTF-8 decode, pending buffer, and validation for the
    // common case of English text (0x21-0x7E maps to itself).
    if (IREE_LIKELY(state->pending_count == 0 && first_byte >= 0x21 &&
                    first_byte <= 0x7E)) {
      if (!iree_tokenizer_byte_level_process_ascii_run(
              token, read_position, output, write_position)) {
        return false;  // Buffer full.
      }
      continue;
    }

    // Slow path: decode input codepoint.
    iree_string_view_t remaining = iree_make_string_view(
        token.data + *read_position, token.size - *read_position);
    iree_host_size_t local_position = 0;
    uint32_t codepoint = iree_unicode_utf8_decode(remaining, &local_position);
    iree_host_size_t codepoint_bytes = local_position;

    // Map codepoint to byte(s).
    if (codepoint >= 0x100 && codepoint <= 0x143) {
      // SHIFTED: reverse mapping table.
      uint8_t mapped =
          iree_tokenizer_byte_level_reverse_mapping[codepoint - 0x100];
      if (!iree_tokenizer_byte_level_handle_mapped_byte(state, mapped, output,
                                                        write_position)) {
        return false;  // Buffer full.
      }
    } else if (iree_tokenizer_byte_level_is_identity(codepoint)) {
      // IDENTITY: codepoint value IS the byte value.
      uint8_t mapped = (uint8_t)codepoint;
      if (!iree_tokenizer_byte_level_handle_mapped_byte(state, mapped, output,
                                                        write_position)) {
        return false;  // Buffer full.
      }
    } else {
      // PASSTHROUGH: flush pending, then handle the codepoint.
      if (!iree_tokenizer_byte_level_flush_pending(state, output,
                                                   write_position)) {
        return false;  // Buffer full.
      }
      if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
        // Invalid input byte - emit replacement char instead of copying
        // garbage.
        static const uint8_t kUtf8ReplacementChar[3] = {0xEF, 0xBF, 0xBD};
        if (*write_position + 3 > output.size) {
          return false;  // Buffer full.
        }
        memcpy(output.data + *write_position, kUtf8ReplacementChar, 3);
        *write_position += 3;
      } else {
        // Valid passthrough codepoint - copy original UTF-8 bytes.
        // Use memmove - source and destination may overlap when used in
        // sequence decoder (same buffer for input and output).
        if (*write_position + codepoint_bytes > output.size) {
          return false;  // Buffer full.
        }
        memmove(output.data + *write_position, token.data + *read_position,
                codepoint_bytes);
        *write_position += codepoint_bytes;
      }
    }
    *read_position += codepoint_bytes;
  }
  return true;  // Token fully consumed.
}

//===----------------------------------------------------------------------===//
// Process Implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_decoder_byte_level_state_process(
    iree_tokenizer_decoder_state_t* base_state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  iree_tokenizer_decoder_byte_level_state_t* state =
      (iree_tokenizer_decoder_byte_level_state_t*)base_state;

  iree_host_size_t strings_consumed = 0;
  iree_host_size_t write_position = 0;

  for (iree_host_size_t i = 0; i < token_strings.count; ++i) {
    iree_string_view_t token = token_strings.values[i];
    iree_host_size_t read_position = (i == 0) ? state->resume_position : 0;

    bool token_consumed = iree_tokenizer_byte_level_process_token(
        state, token, &read_position, output, &write_position);

    if (!token_consumed) {
      // Buffer full mid-token - save position for resume.
      state->resume_position = read_position;
      break;
    }
    state->resume_position = 0;
    strings_consumed++;
  }

  *out_strings_consumed = strings_consumed;
  *out_bytes_written = write_position;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_byte_level_state_finalize(
    iree_tokenizer_decoder_state_t* base_state,
    iree_mutable_string_view_t output, iree_host_size_t* out_written) {
  iree_tokenizer_decoder_byte_level_state_t* state =
      (iree_tokenizer_decoder_byte_level_state_t*)base_state;

  iree_host_size_t written = 0;
  if (state->pending_count > 0) {
    // Any pending bytes at end of stream are incomplete -> U+FFFD.
    // If buffer is too small, preserve pending so caller can retry.
    if (output.size >= 3) {
      output.data[0] = (char)0xEF;
      output.data[1] = (char)0xBF;
      output.data[2] = (char)0xBD;
      written = 3;
      state->pending_count = 0;
    }
  }
  *out_written = written;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_byte_level_state_has_pending(
    const iree_tokenizer_decoder_state_t* base_state) {
  const iree_tokenizer_decoder_byte_level_state_t* state =
      (const iree_tokenizer_decoder_byte_level_state_t*)base_state;
  return state->pending_count > 0;
}

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_byte_level_vtable = {
        .destroy = iree_tokenizer_decoder_byte_level_destroy,
        .state_initialize = iree_tokenizer_decoder_byte_level_state_initialize,
        .state_deinitialize =
            iree_tokenizer_decoder_byte_level_state_deinitialize,
        .state_process = iree_tokenizer_decoder_byte_level_state_process,
        .state_finalize = iree_tokenizer_decoder_byte_level_state_finalize,
        .state_has_pending =
            iree_tokenizer_decoder_byte_level_state_has_pending,
        .flags = IREE_TOKENIZER_DECODER_FLAG_SHRINKING,
};
