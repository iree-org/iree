// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/bert.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// BERT Normalizer Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_normalizer_bert_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
  iree_tokenizer_bert_normalizer_flags_t flags;
} iree_tokenizer_normalizer_bert_t;

// Maximum bytes a single input codepoint can expand to in the BERT pipeline:
//   - CJK spacing: adds 2 bytes (" X ")
//   - NFD decomposition: up to 4 codepoints output
//   - Lowercase: each codepoint can become 2 codepoints (İ → i + dot)
//   - UTF-8 encoding: each codepoint up to 4 bytes
// Conservative max: 4 decomposed * 2 lowercase * 4 bytes = 32 bytes
// Plus 2 bytes for CJK spaces = 34 bytes, round up to 48 for safety.
#define IREE_BERT_MAX_CODEPOINT_OUTPUT 48

// State for streaming BERT processing.
// Handles pending output from codepoint expansion.
typedef struct iree_tokenizer_normalizer_bert_state_t {
  iree_tokenizer_normalizer_state_t base;
  // Cached flags for fast access.
  iree_tokenizer_bert_normalizer_flags_t flags;
  // Pending output bytes from expansion that didn't fit in output buffer.
  uint8_t pending_output[IREE_BERT_MAX_CODEPOINT_OUTPUT];
  uint8_t pending_output_count;
  uint8_t pending_output_position;
} iree_tokenizer_normalizer_bert_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_bert_vtable;

//===----------------------------------------------------------------------===//
// HuggingFace-Compatible Character Classification
//===----------------------------------------------------------------------===//

// Returns true if byte is ASCII (high bit clear).
static inline bool iree_tokenizer_bert_is_ascii(uint8_t byte) {
  return (byte & 0x80) == 0;
}

// HuggingFace-compatible whitespace check.
// Treats \t, \n, \r as whitespace (NOT control characters).
static inline bool iree_tokenizer_bert_is_whitespace(uint32_t codepoint) {
  // These are technically control characters but HF treats them as whitespace.
  if (codepoint == '\t' || codepoint == '\n' || codepoint == '\r') {
    return true;
  }
  return iree_unicode_is_whitespace(codepoint);
}

// HuggingFace-compatible control character check.
// Returns true for "other" category (Cc/Cf/Cn/Co) EXCEPT \t, \n, \r.
static inline bool iree_tokenizer_bert_is_control(uint32_t codepoint) {
  // \t, \n, \r are treated as whitespace, not control.
  if (codepoint == '\t' || codepoint == '\n' || codepoint == '\r') {
    return false;
  }
  // HuggingFace uses is_other() which includes Cc, Cf, Cn, Co.
  return iree_unicode_is_other(codepoint);
}

// HuggingFace-compatible Chinese character check.
// This matches HuggingFace's is_chinese_char() ranges exactly.
//
// Note: HuggingFace uses 0x2B920 for the Extension E/F start, but the actual
// Unicode block starts at 0x2B820. We use 0x2B820 (correct per Unicode spec).
// This difference affects 256 codepoints in a rarely-used extension block.
static inline bool iree_tokenizer_bert_is_chinese_char(uint32_t codepoint) {
  return iree_unicode_is_han(codepoint);
}

//===----------------------------------------------------------------------===//
// Pending Output Management
//===----------------------------------------------------------------------===//

// Emits pending output bytes to the output buffer.
// Returns number of bytes written.
static iree_host_size_t iree_tokenizer_bert_emit_pending(
    iree_tokenizer_normalizer_bert_state_t* state, uint8_t* output,
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

// Appends a byte to the pending output buffer.
static inline void iree_tokenizer_bert_append_byte(
    iree_tokenizer_normalizer_bert_state_t* state, uint8_t byte) {
  IREE_ASSERT(state->pending_output_count < IREE_BERT_MAX_CODEPOINT_OUTPUT);
  state->pending_output[state->pending_output_count++] = byte;
}

// Appends a codepoint (encoded as UTF-8) to the pending output buffer.
static inline void iree_tokenizer_bert_append_codepoint(
    iree_tokenizer_normalizer_bert_state_t* state, uint32_t codepoint) {
  int encoded = iree_unicode_utf8_encode(
      codepoint, (char*)(state->pending_output + state->pending_output_count));
  if (encoded > 0) {
    state->pending_output_count += (uint8_t)encoded;
  }
}

//===----------------------------------------------------------------------===//
// BERT Transformation Pipeline
//===----------------------------------------------------------------------===//

// Processes a single codepoint through the BERT transformation pipeline.
// Output is accumulated in state->pending_output.
// Order: clean_text → handle_chinese_chars → strip_accents → lowercase
static void iree_tokenizer_bert_process_codepoint(
    iree_tokenizer_normalizer_bert_state_t* state, uint32_t codepoint) {
  const iree_tokenizer_bert_normalizer_flags_t flags = state->flags;

  // Reset pending output for this codepoint.
  state->pending_output_count = 0;
  state->pending_output_position = 0;

  //-------------------------------------------------------------------------
  // Phase 1: clean_text (character filtering and mapping)
  //-------------------------------------------------------------------------
  if (flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT) {
    // Remove null and replacement char explicitly.
    if (codepoint == 0 || codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
      return;  // Filtered out, no output.
    }

    // Check whitespace FIRST - \t, \n, \r are whitespace, NOT control chars.
    if (iree_tokenizer_bert_is_whitespace(codepoint)) {
      codepoint = ' ';  // Map all whitespace to space.
    } else if (iree_tokenizer_bert_is_control(codepoint)) {
      return;  // Control chars (excluding whitespace) are removed.
    }
  }

  //-------------------------------------------------------------------------
  // Phase 2: handle_chinese_chars (leading space before CJK)
  //-------------------------------------------------------------------------
  bool is_chinese = false;
  if ((flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS) &&
      iree_tokenizer_bert_is_chinese_char(codepoint)) {
    is_chinese = true;
    iree_tokenizer_bert_append_byte(state, ' ');
  }

  //-------------------------------------------------------------------------
  // Phase 3 & 4: strip_accents + lowercase
  //-------------------------------------------------------------------------
  if (flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS) {
    // NFD decompose the codepoint.
    uint32_t decomposed[4];
    iree_host_size_t decomposed_count =
        iree_unicode_decompose(codepoint, decomposed);

    // Process each decomposed codepoint.
    for (iree_host_size_t i = 0; i < decomposed_count; ++i) {
      uint32_t cp = decomposed[i];

      // Skip Nonspacing Marks (Mn category) - this is the accent stripping.
      // HuggingFace uses is_mark_nonspacing() which is specifically Mn,
      // NOT Mc (Spacing Combining) or Me (Enclosing).
      if (iree_unicode_is_mark_nonspacing(cp)) {
        continue;
      }

      // Apply lowercase if enabled.
      if (flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE) {
        uint32_t lower[2];
        iree_host_size_t lower_count = iree_unicode_to_lower(cp, lower);
        for (iree_host_size_t j = 0; j < lower_count; ++j) {
          iree_tokenizer_bert_append_codepoint(state, lower[j]);
        }
      } else {
        iree_tokenizer_bert_append_codepoint(state, cp);
      }
    }
  } else if (flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE) {
    // Lowercase only (no NFD decomposition needed).
    uint32_t lower[2];
    iree_host_size_t lower_count = iree_unicode_to_lower(codepoint, lower);
    for (iree_host_size_t j = 0; j < lower_count; ++j) {
      iree_tokenizer_bert_append_codepoint(state, lower[j]);
    }
  } else {
    // No strip_accents or lowercase - emit codepoint as-is.
    iree_tokenizer_bert_append_codepoint(state, codepoint);
  }

  //-------------------------------------------------------------------------
  // Phase 2 continued: handle_chinese_chars (trailing space after CJK)
  //-------------------------------------------------------------------------
  if (is_chinese) {
    iree_tokenizer_bert_append_byte(state, ' ');
  }
}

//===----------------------------------------------------------------------===//
// VTable Implementation
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_normalizer_bert_allocate(
    iree_tokenizer_bert_normalizer_flags_t flags, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  iree_tokenizer_normalizer_bert_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_bert_vtable,
      sizeof(iree_tokenizer_normalizer_bert_state_t));
  normalizer->allocator = allocator;
  normalizer->flags = flags;

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_bert_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_bert_t* self =
      (iree_tokenizer_normalizer_bert_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_bert_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_normalizer_bert_t* bert_normalizer =
      (const iree_tokenizer_normalizer_bert_t*)normalizer;
  iree_tokenizer_normalizer_bert_state_t* state =
      (iree_tokenizer_normalizer_bert_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  // Cache flags in state for fast access during processing.
  state->flags = bert_normalizer->flags;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_bert_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_bert_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  (void)flags;  // BERT normalizer is stateless within codepoints.

  iree_tokenizer_normalizer_bert_state_t* state =
      (iree_tokenizer_normalizer_bert_state_t*)base_state;

  const uint8_t* in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // First, emit any pending output from previous call.
  if (state->pending_output_count > state->pending_output_position) {
    iree_host_size_t written = iree_tokenizer_bert_emit_pending(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;
    if (state->pending_output_count > state->pending_output_position) {
      // Output buffer full, couldn't emit all pending.
      *out_consumed = 0;
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Check if all flags are disabled - fast passthrough path.
  if (state->flags == IREE_TOKENIZER_BERT_NORMALIZER_FLAG_NONE) {
    iree_host_size_t copy_size = (iree_host_size_t)(in_end - in_ptr);
    iree_host_size_t output_space = (iree_host_size_t)(out_end - out_ptr);
    if (copy_size > output_space) {
      copy_size = output_space;
    }
    memcpy(out_ptr, in_ptr, copy_size);
    *out_consumed = copy_size;
    *out_written =
        (iree_host_size_t)((out_ptr + copy_size) - (uint8_t*)output.data);
    return iree_ok_status();
  }

  // Main processing loop.
  while (in_ptr < in_end && out_ptr < out_end) {
    // ASCII fast path: for common ASCII text with simple flags.
    // Only applies when clean_text is off or character is printable ASCII.
    if (iree_tokenizer_bert_is_ascii(*in_ptr)) {
      uint32_t codepoint = *in_ptr;

      // Process through pipeline.
      iree_tokenizer_bert_process_codepoint(state, codepoint);

      // Try to emit pending output.
      if (state->pending_output_count > 0) {
        iree_host_size_t written = iree_tokenizer_bert_emit_pending(
            state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
        out_ptr += written;

        if (state->pending_output_count > state->pending_output_position) {
          // Output buffer full - consumed the input byte but couldn't emit all.
          ++in_ptr;
          break;
        }
      }

      ++in_ptr;
      continue;
    }

    // Non-ASCII byte encountered - decode full codepoint.
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

    // Process through pipeline.
    iree_tokenizer_bert_process_codepoint(state, codepoint);

    // Try to emit pending output.
    if (state->pending_output_count > 0) {
      iree_host_size_t written = iree_tokenizer_bert_emit_pending(
          state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
      out_ptr += written;

      if (state->pending_output_count > state->pending_output_position) {
        // Output buffer full - consumed the input but couldn't emit all.
        in_ptr += position;
        break;
      }
    }

    in_ptr += position;
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_bert_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_bert_state_t* state =
      (iree_tokenizer_normalizer_bert_state_t*)base_state;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Emit any pending output.
  if (state->pending_output_count > state->pending_output_position) {
    iree_host_size_t written = iree_tokenizer_bert_emit_pending(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += written;
  }

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_bert_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_bert_state_t* state =
      (const iree_tokenizer_normalizer_bert_state_t*)base_state;
  return state->pending_output_count > state->pending_output_position;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_bert_vtable = {
        .destroy = iree_tokenizer_normalizer_bert_destroy,
        .state_initialize = iree_tokenizer_normalizer_bert_state_initialize,
        .state_deinitialize = iree_tokenizer_normalizer_bert_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_bert_state_process,
        .state_finalize = iree_tokenizer_normalizer_bert_state_finalize,
        .state_has_pending = iree_tokenizer_normalizer_bert_state_has_pending,
};
