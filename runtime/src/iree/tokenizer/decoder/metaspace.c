// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/metaspace.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

// Default replacement character: U+2581 (LOWER ONE EIGHTH BLOCK).
#define IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT 0x2581

//===----------------------------------------------------------------------===//
// Metaspace Decoder Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_decoder_metaspace_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
  // Replacement codepoint (default U+2581).
  uint32_t replacement_codepoint;
  // UTF-8 encoding of the replacement codepoint.
  char replacement_utf8[4];
  uint8_t replacement_utf8_length;
  // Prepend scheme for first token handling.
  iree_tokenizer_decoder_metaspace_prepend_scheme_t prepend_scheme;
} iree_tokenizer_decoder_metaspace_t;

// State is cache-line friendly: all data needed for hot path is here.
// process() never touches the decoder pointer after initialization.
typedef struct iree_tokenizer_decoder_metaspace_state_t {
  iree_tokenizer_decoder_state_t base;
  // Copied from decoder at init time for cache locality.
  char replacement_utf8[4];
  uint8_t replacement_utf8_length;
  uint8_t prepend_scheme;
  // Whether we've produced any output yet.
  bool seen_first_output;
} iree_tokenizer_decoder_metaspace_state_t;

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_metaspace_vtable;

iree_status_t iree_tokenizer_decoder_metaspace_allocate(
    uint32_t replacement_codepoint,
    iree_tokenizer_decoder_metaspace_prepend_scheme_t prepend_scheme,
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (replacement_codepoint == 0) {
    replacement_codepoint = IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT;
  }

  iree_tokenizer_decoder_metaspace_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  iree_tokenizer_decoder_capability_t capabilities =
      IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS;
  if (prepend_scheme != IREE_TOKENIZER_DECODER_METASPACE_PREPEND_NEVER) {
    capabilities |= IREE_TOKENIZER_DECODER_CAPABILITY_POSITION_SENSITIVE;
  }
  iree_tokenizer_decoder_initialize(
      &decoder->base, &iree_tokenizer_decoder_metaspace_vtable,
      sizeof(iree_tokenizer_decoder_metaspace_state_t), capabilities);
  decoder->allocator = allocator;
  decoder->replacement_codepoint = replacement_codepoint;
  decoder->prepend_scheme = prepend_scheme;

  // Pre-encode the replacement character to UTF-8.
  iree_status_t status = iree_ok_status();
  int encoded_length = iree_unicode_utf8_encode(replacement_codepoint,
                                                decoder->replacement_utf8);
  if (encoded_length <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid replacement codepoint");
  } else {
    decoder->replacement_utf8_length = (uint8_t)encoded_length;
  }

  if (iree_status_is_ok(status)) {
    *out_decoder = &decoder->base;
  } else {
    iree_allocator_free(allocator, decoder);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_tokenizer_decoder_metaspace_destroy(
    iree_tokenizer_decoder_t* decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_metaspace_t* self =
      (iree_tokenizer_decoder_metaspace_t*)decoder;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_metaspace_state_initialize(
    const iree_tokenizer_decoder_t* decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_decoder_metaspace_t* self =
      (const iree_tokenizer_decoder_metaspace_t*)decoder;
  iree_tokenizer_decoder_metaspace_state_t* state =
      (iree_tokenizer_decoder_metaspace_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.decoder = decoder;

  // Copy config to state for cache-friendly hot path.
  memcpy(state->replacement_utf8, self->replacement_utf8,
         self->replacement_utf8_length);
  state->replacement_utf8_length = self->replacement_utf8_length;
  state->prepend_scheme = (uint8_t)self->prepend_scheme;
  state->seen_first_output = false;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_metaspace_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Processes a single token, replacing metaspace characters with spaces.
// Returns the number of bytes written to output.
// Updates |*token_fully_consumed| to indicate if the entire token was
// processed.
static iree_host_size_t iree_tokenizer_decoder_metaspace_process_token(
    iree_tokenizer_decoder_metaspace_state_t* state, iree_string_view_t token,
    iree_mutable_string_view_t output, iree_host_size_t output_position,
    bool* token_fully_consumed) {
  const char* replacement_utf8 = state->replacement_utf8;
  const iree_host_size_t replacement_length = state->replacement_utf8_length;
  iree_host_size_t bytes_written = 0;
  iree_host_size_t token_position = 0;

  *token_fully_consumed = false;

  while (token_position < token.size) {
    // Check if we're at a metaspace character.
    bool is_metaspace = false;
    if (token_position + replacement_length <= token.size) {
      is_metaspace = (memcmp(token.data + token_position, replacement_utf8,
                             replacement_length) == 0);
    }

    if (is_metaspace) {
      // Determine if we should strip this metaspace.
      bool strip_metaspace = false;
      if (!state->seen_first_output && token_position == 0 &&
          bytes_written == 0) {
        switch (state->prepend_scheme) {
          case IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS:
            strip_metaspace = true;
            break;
          case IREE_TOKENIZER_DECODER_METASPACE_PREPEND_NEVER:
            strip_metaspace = false;
            break;
          case IREE_TOKENIZER_DECODER_METASPACE_PREPEND_FIRST:
            strip_metaspace = true;
            break;
          default:
            IREE_ASSERT(false && "invalid metaspace prepend scheme");
            break;
        }
      }

      if (strip_metaspace) {
        token_position += replacement_length;
        state->seen_first_output = true;
      } else {
        // Replace metaspace with space.
        iree_host_size_t required_size = 0;
        if (!iree_host_size_checked_add(output_position, bytes_written,
                                        &required_size) ||
            !iree_host_size_checked_add(required_size, 1, &required_size) ||
            required_size > output.size) {
          return bytes_written;  // Overflow or no room.
        }
        output.data[output_position + bytes_written] = ' ';
        bytes_written++;
        token_position += replacement_length;
        state->seen_first_output = true;
      }
    } else {
      // Find the next metaspace or end of token.
      iree_host_size_t copy_start = token_position;
      while (token_position < token.size) {
        if (token_position + replacement_length <= token.size &&
            memcmp(token.data + token_position, replacement_utf8,
                   replacement_length) == 0) {
          break;
        }
        iree_host_size_t cp_length = iree_unicode_utf8_sequence_length(
            (uint8_t)token.data[token_position]);
        if (token_position + cp_length > token.size) {
          cp_length = token.size - token_position;
        }
        token_position += cp_length;
      }

      iree_host_size_t copy_length = token_position - copy_start;
      // Compute available space with overflow check. If overflow, treat as 0.
      iree_host_size_t used = 0;
      iree_host_size_t available = 0;
      if (iree_host_size_checked_add(output_position, bytes_written, &used) &&
          used <= output.size) {
        available = output.size - used;
      }
      if (copy_length > available) {
        copy_length = available;
        token_position = copy_start + copy_length;  // Partial.
      }

      if (copy_length > 0) {
        // Use memmove - source and destination may overlap when used in
        // sequence decoder (same buffer for input and output).
        memmove(output.data + output_position + bytes_written,
                token.data + copy_start, copy_length);
        bytes_written += copy_length;
        state->seen_first_output = true;
      }

      // Only return early if buffer is full (or if position calculation
      // overflows, which implies we've somehow exceeded addressable memory).
      iree_host_size_t current_position = 0;
      if (!iree_host_size_checked_add(output_position, bytes_written,
                                      &current_position) ||
          current_position >= output.size) {
        return bytes_written;
      }
    }
  }

  *token_fully_consumed = true;
  return bytes_written;
}

static iree_status_t iree_tokenizer_decoder_metaspace_state_process(
    iree_tokenizer_decoder_state_t* base_state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  iree_tokenizer_decoder_metaspace_state_t* state =
      (iree_tokenizer_decoder_metaspace_state_t*)base_state;

  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  for (iree_host_size_t i = 0; i < token_strings.count; ++i) {
    if (bytes_written >= output.size) break;

    bool token_fully_consumed = false;
    iree_host_size_t written = iree_tokenizer_decoder_metaspace_process_token(
        state, token_strings.values[i], output, bytes_written,
        &token_fully_consumed);
    bytes_written += written;

    if (token_fully_consumed) {
      strings_consumed++;
    } else {
      break;  // Couldn't finish this token - buffer full.
    }
  }

  *out_strings_consumed = strings_consumed;
  *out_bytes_written = bytes_written;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_metaspace_state_finalize(
    iree_tokenizer_decoder_state_t* state, iree_mutable_string_view_t output,
    iree_host_size_t* out_written) {
  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_metaspace_state_has_pending(
    const iree_tokenizer_decoder_state_t* state) {
  return false;
}

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_metaspace_vtable = {
        .destroy = iree_tokenizer_decoder_metaspace_destroy,
        .state_initialize = iree_tokenizer_decoder_metaspace_state_initialize,
        .state_deinitialize =
            iree_tokenizer_decoder_metaspace_state_deinitialize,
        .state_process = iree_tokenizer_decoder_metaspace_state_process,
        .state_finalize = iree_tokenizer_decoder_metaspace_state_finalize,
        .state_has_pending = iree_tokenizer_decoder_metaspace_state_has_pending,
        .flags = IREE_TOKENIZER_DECODER_FLAG_SHRINKING,
};
