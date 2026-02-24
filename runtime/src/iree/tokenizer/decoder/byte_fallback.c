// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/byte_fallback.h"

#include <stdbool.h>
#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// ByteFallback Decoder Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_decoder_byte_fallback_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
} iree_tokenizer_decoder_byte_fallback_t;

// Streaming state for ByteFallback decoder.
typedef struct iree_tokenizer_decoder_byte_fallback_state_t {
  iree_tokenizer_decoder_state_t base;
  // Pending bytes accumulating a UTF-8 sequence.
  uint8_t pending_bytes[4];
  // Number of bytes in pending_bytes (0-4).
  uint8_t pending_count;
  // Expected total bytes for current UTF-8 sequence (0 if not started).
  uint8_t expected_length;
} iree_tokenizer_decoder_byte_fallback_state_t;

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_byte_fallback_vtable;

iree_status_t iree_tokenizer_decoder_byte_fallback_allocate(
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_decoder_byte_fallback_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  iree_tokenizer_decoder_initialize(
      &decoder->base, &iree_tokenizer_decoder_byte_fallback_vtable,
      sizeof(iree_tokenizer_decoder_byte_fallback_state_t),
      IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS_EXCEPT_BYTE_TOKENS);
  decoder->allocator = allocator;

  *out_decoder = &decoder->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_byte_fallback_destroy(
    iree_tokenizer_decoder_t* decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_byte_fallback_t* self =
      (iree_tokenizer_decoder_byte_fallback_t*)decoder;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_byte_fallback_state_initialize(
    const iree_tokenizer_decoder_t* decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_byte_fallback_state_t* state =
      (iree_tokenizer_decoder_byte_fallback_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.decoder = decoder;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_byte_fallback_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Parses a token matching the `<0xHH>` pattern.
// Returns true if the token matches, with the byte value in |out_byte|.
bool iree_tokenizer_decoder_byte_fallback_parse_byte_token(
    iree_string_view_t token, uint8_t* out_byte) {
  // Pattern: `<0xHH>` where H is hex digit. Exactly 6 characters.
  if (token.size != 6) return false;
  if (token.data[0] != '<') return false;
  if (token.data[1] != '0') return false;
  if (token.data[2] != 'x' && token.data[2] != 'X') return false;
  if (token.data[5] != '>') return false;

  // Extract the two hex digits and parse.
  iree_string_view_t hex_digits = iree_make_string_view(&token.data[3], 2);
  uint32_t value = 0;
  if (!iree_string_view_atoi_uint32_base(hex_digits, 16, &value)) {
    return false;
  }
  if (value > 255) return false;

  *out_byte = (uint8_t)value;
  return true;
}

// Writes replacement characters to output for each pending byte.
// Returns bytes written to output, or 0 if not enough space.
static iree_host_size_t
iree_tokenizer_decoder_byte_fallback_flush_as_replacement(
    iree_tokenizer_decoder_byte_fallback_state_t* state,
    iree_mutable_string_view_t output, iree_host_size_t position) {
  // Each pending byte becomes one replacement character (3 bytes UTF-8).
  static const iree_host_size_t kReplacementLength = 3;  // U+FFFD = EF BF BD
  iree_host_size_t written = 0;

  while (state->pending_count > 0) {
    if (position + written + kReplacementLength > output.size) {
      // Not enough space - leave remaining pending bytes for later.
      break;
    }
    // Encode U+FFFD replacement character.
    int encoded_length = iree_unicode_utf8_encode(
        IREE_UNICODE_REPLACEMENT_CHAR, output.data + position + written);
    if (encoded_length <= 0) break;  // Should not happen.
    written += (iree_host_size_t)encoded_length;

    // Shift pending bytes left.
    for (uint8_t i = 0; i < state->pending_count - 1; i++) {
      state->pending_bytes[i] = state->pending_bytes[i + 1];
    }
    state->pending_count--;
  }

  if (state->pending_count == 0) {
    state->expected_length = 0;
  }
  return written;
}

// Validates and emits the accumulated bytes as UTF-8.
// Sets |out_is_valid| to indicate whether the sequence is valid UTF-8.
// Returns bytes written (0 if buffer full or invalid).
// Caller should:
// - If return > 0: success, sequence emitted
// - If return == 0 && *out_is_valid: buffer full, preserve pending for retry
// - If return == 0 && !*out_is_valid: invalid, emit replacements
static iree_host_size_t
iree_tokenizer_decoder_byte_fallback_emit_valid_sequence(
    iree_tokenizer_decoder_byte_fallback_state_t* state,
    iree_mutable_string_view_t output, iree_host_size_t position,
    bool* out_is_valid) {
  // Create a string view from pending bytes and validate via utf8_decode.
  iree_string_view_t pending = iree_make_string_view(
      (const char*)state->pending_bytes, state->pending_count);
  iree_host_size_t decode_position = 0;
  uint32_t codepoint = iree_unicode_utf8_decode(pending, &decode_position);

  // If decode consumed all bytes and didn't return replacement, it's valid.
  bool is_valid = (decode_position == state->pending_count &&
                   codepoint != IREE_UNICODE_REPLACEMENT_CHAR);
  *out_is_valid = is_valid;

  if (is_valid) {
    // Valid sequence - check buffer space first.
    if (position + state->pending_count > output.size) {
      return 0;  // Buffer full - pending preserved for retry.
    }
    memcpy(output.data + position, state->pending_bytes, state->pending_count);
    iree_host_size_t written = state->pending_count;
    state->pending_count = 0;
    state->expected_length = 0;
    return written;
  }

  // Invalid sequence - return 0 to signal caller should emit replacements.
  return 0;
}

// Starts a new UTF-8 sequence with the given lead byte.
// Handles ASCII (emits directly), invalid leads (emits replacement), and
// multi-byte leads (stores as pending).
// Returns:
//   >0: bytes written to output
//   0: multi-byte sequence started, nothing written yet
//   -1: no buffer space, state preserved for retry
static int iree_tokenizer_decoder_byte_fallback_start_sequence(
    iree_tokenizer_decoder_byte_fallback_state_t* state, uint8_t byte_value,
    iree_mutable_string_view_t output, iree_host_size_t* write_position) {
  static const iree_host_size_t kReplacementLength = 3;

  state->expected_length =
      (uint8_t)iree_unicode_utf8_sequence_length(byte_value);
  state->pending_bytes[0] = byte_value;
  state->pending_count = 1;

  // Invalid lead byte (continuation byte used as lead).
  if (state->expected_length == 1 && (byte_value & 0x80) != 0) {
    if (*write_position + kReplacementLength > output.size) {
      return -1;  // No room for replacement.
    }
    iree_host_size_t flushed =
        iree_tokenizer_decoder_byte_fallback_flush_as_replacement(
            state, output, *write_position);
    *write_position += flushed;
    return (int)flushed;
  }

  // Single-byte ASCII.
  if (state->expected_length == 1) {
    if (*write_position + 1 > output.size) {
      return -1;  // No room.
    }
    output.data[(*write_position)++] = (char)byte_value;
    state->pending_count = 0;
    state->expected_length = 0;
    return 1;
  }

  // Multi-byte sequence started - nothing to write yet.
  return 0;
}

static iree_status_t iree_tokenizer_decoder_byte_fallback_state_process(
    iree_tokenizer_decoder_state_t* base_state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  iree_tokenizer_decoder_byte_fallback_state_t* state =
      (iree_tokenizer_decoder_byte_fallback_state_t*)base_state;

  static const iree_host_size_t kReplacementLength = 3;
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  for (iree_host_size_t i = 0; i < token_strings.count; ++i) {
    iree_string_view_t token = token_strings.values[i];

    uint8_t byte_value;
    if (iree_tokenizer_decoder_byte_fallback_parse_byte_token(token,
                                                              &byte_value)) {
      // Byte token - accumulate for UTF-8 validation.
      if (state->pending_count == 0) {
        // No pending - start new sequence.
        int result = iree_tokenizer_decoder_byte_fallback_start_sequence(
            state, byte_value, output, &bytes_written);
        if (result < 0) break;  // No buffer space.
        strings_consumed++;
        continue;
      }

      // Have pending bytes - expecting continuation.
      if ((byte_value & 0xC0) != 0x80) {
        // Not a continuation - flush pending, then start new sequence.
        bytes_written +=
            iree_tokenizer_decoder_byte_fallback_flush_as_replacement(
                state, output, bytes_written);
        if (state->pending_count > 0) break;  // Partial flush, stop here.

        int result = iree_tokenizer_decoder_byte_fallback_start_sequence(
            state, byte_value, output, &bytes_written);
        if (result < 0) break;
        strings_consumed++;
        continue;
      }

      // Valid continuation byte.
      if (state->pending_count + 1 == state->expected_length) {
        // Would complete sequence - tentatively accumulate and try to emit.
        state->pending_bytes[state->pending_count++] = byte_value;

        bool is_valid = false;
        iree_host_size_t emitted =
            iree_tokenizer_decoder_byte_fallback_emit_valid_sequence(
                state, output, bytes_written, &is_valid);
        if (emitted > 0) {
          bytes_written += emitted;
        } else if (is_valid) {
          // Valid but buffer full - undo and retry later.
          state->pending_count--;
          break;
        } else {
          // Invalid sequence - flush as replacements.
          iree_host_size_t flushed =
              iree_tokenizer_decoder_byte_fallback_flush_as_replacement(
                  state, output, bytes_written);
          if (flushed == 0) {
            state->pending_count--;
            break;
          }
          bytes_written += flushed;
        }
      } else if (state->pending_count < sizeof(state->pending_bytes)) {
        // Still incomplete - just accumulate.
        state->pending_bytes[state->pending_count++] = byte_value;
      } else {
        // Buffer full but sequence not complete (malformed expected_length).
        // Flush pending as replacements and start a new sequence.
        bytes_written +=
            iree_tokenizer_decoder_byte_fallback_flush_as_replacement(
                state, output, bytes_written);
        if (state->pending_count > 0) break;  // Partial flush, retry later.
        int result = iree_tokenizer_decoder_byte_fallback_start_sequence(
            state, byte_value, output, &bytes_written);
        if (result < 0) break;  // No buffer space.
      }

      strings_consumed++;
    } else {
      // Non-byte token - flush pending and pass through.
      if (state->pending_count > 0) {
        iree_host_size_t replacement_size =
            state->pending_count * kReplacementLength;
        if (bytes_written + replacement_size > output.size) break;
        bytes_written +=
            iree_tokenizer_decoder_byte_fallback_flush_as_replacement(
                state, output, bytes_written);
      }

      // Pass through the token. Use memmove - source and destination may
      // overlap when used in sequence decoder (same buffer for input/output).
      if (bytes_written + token.size > output.size) {
        break;  // No room for this token.
      }
      if (token.size > 0) {
        memmove(output.data + bytes_written, token.data, token.size);
      }
      bytes_written += token.size;
      strings_consumed++;
    }
  }

  *out_strings_consumed = strings_consumed;
  *out_bytes_written = bytes_written;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_byte_fallback_state_finalize(
    iree_tokenizer_decoder_state_t* base_state,
    iree_mutable_string_view_t output, iree_host_size_t* out_written) {
  iree_tokenizer_decoder_byte_fallback_state_t* state =
      (iree_tokenizer_decoder_byte_fallback_state_t*)base_state;

  static const iree_host_size_t kReplacementLength = 3;
  iree_host_size_t written = 0;

  // Flush any pending bytes as replacement characters.
  if (state->pending_count > 0) {
    iree_host_size_t replacement_size =
        state->pending_count * kReplacementLength;
    if (replacement_size > output.size) {
      *out_written = 0;
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small for pending bytes");
    }
    written = iree_tokenizer_decoder_byte_fallback_flush_as_replacement(
        state, output, 0);
  }

  *out_written = written;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_byte_fallback_state_has_pending(
    const iree_tokenizer_decoder_state_t* base_state) {
  const iree_tokenizer_decoder_byte_fallback_state_t* state =
      (const iree_tokenizer_decoder_byte_fallback_state_t*)base_state;
  return state->pending_count > 0;
}

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_byte_fallback_vtable = {
        .destroy = iree_tokenizer_decoder_byte_fallback_destroy,
        .state_initialize =
            iree_tokenizer_decoder_byte_fallback_state_initialize,
        .state_deinitialize =
            iree_tokenizer_decoder_byte_fallback_state_deinitialize,
        .state_process = iree_tokenizer_decoder_byte_fallback_state_process,
        .state_finalize = iree_tokenizer_decoder_byte_fallback_state_finalize,
        .state_has_pending =
            iree_tokenizer_decoder_byte_fallback_state_has_pending,
        .flags = IREE_TOKENIZER_DECODER_FLAG_SHRINKING,
};
