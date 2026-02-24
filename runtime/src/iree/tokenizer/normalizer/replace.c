// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/replace.h"

#include <string.h>

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_replace_single_vtable;
static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_replace_multi_vtable;

//===----------------------------------------------------------------------===//
// Replace Normalizer (Shared)
//===----------------------------------------------------------------------===//

// Normalizer struct shared by both single-byte and multi-byte implementations.
// Pattern and content data follow immediately after this struct.
typedef struct iree_tokenizer_normalizer_replace_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
  uint8_t pattern_length;
  uint8_t content_length;
  // Trailing data: [pattern][content]
} iree_tokenizer_normalizer_replace_t;

// Maximum overlap buffer size = max pattern length - 1.
#define IREE_TOKENIZER_REPLACE_MAX_OVERLAP \
  (IREE_TOKENIZER_REPLACE_MAX_PATTERN - 1)

//===----------------------------------------------------------------------===//
// Single-Byte State Structure
//===----------------------------------------------------------------------===//
// Optimized path for pattern_length == 1. Zero buffering for pattern matching;
// only tracks partial content emission when output is smaller than content.

typedef struct iree_tokenizer_replace_single_state_t {
  iree_tokenizer_normalizer_state_t base;
  // Control fields (hot, first cache line).
  uint8_t target_byte;
  uint8_t content_length;
  // 0 in 98% of calls.
  uint8_t content_emitted;
  uint8_t _pad;
  // Points to normalizer trailing data.
  const uint8_t* content;
} iree_tokenizer_replace_single_state_t;

//===----------------------------------------------------------------------===//
// Multi-Byte State Structure
//===----------------------------------------------------------------------===//
// Handles pattern_length > 1 using overlap buffer for cross-chunk matching
// and memchr + memcmp for SIMD-accelerated scanning.

typedef struct iree_tokenizer_replace_multi_state_t {
  iree_tokenizer_normalizer_state_t base;

  // === Control fields (hot, first cache line) ===
  // Points to normalizer trailing data.
  const uint8_t* pattern;
  // Points to normalizer trailing data.
  const uint8_t* content;
  uint8_t pattern_length;  // 2-32.
  uint8_t content_length;  // 0-32.
  // Current overlap buffer fill (0 to pattern_length-1).
  uint8_t overlap_length;
  // 0 in 98% of calls (fast path).
  uint8_t pending_offset;
  // 0 in 98% of calls (fast path).
  uint8_t pending_length;
  uint8_t _pad[3];

  // === Buffers (rarely accessed in hot path) ===
  uint8_t overlap_buffer[IREE_TOKENIZER_REPLACE_MAX_OVERLAP];  // 31 bytes.
  uint8_t pending_buffer[IREE_TOKENIZER_REPLACE_MAX_CONTENT];  // 32 bytes.
} iree_tokenizer_replace_multi_state_t;

//===----------------------------------------------------------------------===//
// Shared Helpers
//===----------------------------------------------------------------------===//

// Returns pointer to pattern data (immediately after struct).
static inline const uint8_t* iree_tokenizer_normalizer_replace_pattern(
    const iree_tokenizer_normalizer_replace_t* normalizer) {
  return (const uint8_t*)(normalizer + 1);
}

// Returns pointer to content data (after pattern).
static inline const uint8_t* iree_tokenizer_normalizer_replace_content(
    const iree_tokenizer_normalizer_replace_t* normalizer) {
  return (const uint8_t*)(normalizer + 1) + normalizer->pattern_length;
}

static void iree_tokenizer_normalizer_replace_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_replace_t* self =
      (iree_tokenizer_normalizer_replace_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_normalizer_replace_allocate(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  // Validate pattern.
  if (pattern.size == 0) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "replace normalizer requires non-empty pattern"));
  }
  if (pattern.size > IREE_TOKENIZER_REPLACE_MAX_PATTERN) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "replace pattern exceeds maximum length (%" PRIhsz
                             " > %d)",
                             pattern.size, IREE_TOKENIZER_REPLACE_MAX_PATTERN));
  }
  if (content.size > IREE_TOKENIZER_REPLACE_MAX_CONTENT) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "replace content exceeds maximum length (%" PRIhsz
                             " > %d)",
                             content.size, IREE_TOKENIZER_REPLACE_MAX_CONTENT));
  }

  // Calculate allocation size with overflow checking.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              iree_sizeof_struct(iree_tokenizer_normalizer_replace_t),
              &total_size, IREE_STRUCT_FIELD(pattern.size, uint8_t, NULL),
              IREE_STRUCT_FIELD(content.size, uint8_t, NULL)));

  iree_tokenizer_normalizer_replace_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&normalizer));

  // Select vtable based on pattern length.
  const iree_tokenizer_normalizer_vtable_t* vtable;
  iree_host_size_t state_size;
  if (pattern.size == 1) {
    vtable = &iree_tokenizer_normalizer_replace_single_vtable;
    state_size = sizeof(iree_tokenizer_replace_single_state_t);
  } else {
    vtable = &iree_tokenizer_normalizer_replace_multi_vtable;
    state_size = sizeof(iree_tokenizer_replace_multi_state_t);
  }

  iree_tokenizer_normalizer_initialize(&normalizer->base, vtable, state_size);
  normalizer->allocator = allocator;
  normalizer->pattern_length = (uint8_t)pattern.size;
  normalizer->content_length = (uint8_t)content.size;

  // Copy pattern and content to trailing storage.
  uint8_t* trailing_data = (uint8_t*)(normalizer + 1);
  memcpy(trailing_data, pattern.data, pattern.size);
  memcpy(trailing_data + pattern.size, content.data, content.size);

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Single-Byte Implementation
//===----------------------------------------------------------------------===//

// Updates partial content emission state after a write.
static inline void iree_tokenizer_replace_single_update_emission(
    iree_tokenizer_replace_single_state_t* state, uint8_t written,
    uint8_t content_length) {
  state->content_emitted = (state->content_emitted + written >= content_length)
                               ? 0
                               : state->content_emitted + written;
}

static iree_status_t iree_tokenizer_replace_single_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_normalizer_replace_t* self =
      (const iree_tokenizer_normalizer_replace_t*)normalizer;
  iree_tokenizer_replace_single_state_t* state =
      (iree_tokenizer_replace_single_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  state->target_byte = *iree_tokenizer_normalizer_replace_pattern(self);
  state->content_length = self->content_length;
  state->content = iree_tokenizer_normalizer_replace_content(self);
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_replace_single_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_replace_single_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  (void)flags;  // Replace has no cross-segment state to reset.

  iree_tokenizer_replace_single_state_t* state =
      (iree_tokenizer_replace_single_state_t*)base_state;

  const uint8_t* IREE_RESTRICT in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* IREE_RESTRICT out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  const uint8_t target_byte = state->target_byte;
  const uint8_t* IREE_RESTRICT content = state->content;
  const uint8_t content_length = state->content_length;

  // Resume partial content emission from previous call.
  if (state->content_emitted > 0) {
    iree_host_size_t remaining = content_length - state->content_emitted;
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    iree_host_size_t to_write =
        remaining < output_available ? remaining : output_available;
    if (to_write > 0) {
      memcpy(out_ptr, content + state->content_emitted, to_write);
      out_ptr += to_write;
    }
    iree_tokenizer_replace_single_update_emission(state, (uint8_t)to_write,
                                                  content_length);
    if (state->content_emitted > 0) {
      *out_consumed = 0;
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Main loop: scan input, replacing target_byte with content.
  while (in_ptr < in_end && out_ptr < out_end) {
    uint8_t byte = *in_ptr;
    if (byte == target_byte) {
      iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
      if (content_length == 0) {
        // Deletion: consume target byte, emit nothing.
        ++in_ptr;
      } else if (output_available >= content_length) {
        memcpy(out_ptr, content, content_length);
        out_ptr += content_length;
        ++in_ptr;
      } else {
        // Partial write.
        memcpy(out_ptr, content, output_available);
        out_ptr += output_available;
        state->content_emitted = (uint8_t)output_available;
        ++in_ptr;
        break;  // Output full.
      }
    } else {
      *out_ptr++ = byte;
      ++in_ptr;
    }
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_replace_single_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_replace_single_state_t* state =
      (iree_tokenizer_replace_single_state_t*)base_state;

  if (state->content_emitted > 0) {
    iree_host_size_t remaining = state->content_length - state->content_emitted;
    iree_host_size_t to_write =
        remaining < output.size ? remaining : output.size;
    if (to_write > 0) {
      memcpy(output.data, state->content + state->content_emitted, to_write);
    }
    iree_tokenizer_replace_single_update_emission(state, (uint8_t)to_write,
                                                  state->content_length);
    *out_written = to_write;
    return iree_ok_status();
  }

  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_replace_single_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_replace_single_state_t* state =
      (const iree_tokenizer_replace_single_state_t*)base_state;
  return state->content_emitted > 0;
}

//===----------------------------------------------------------------------===//
// Multi-Byte Implementation
//===----------------------------------------------------------------------===//

// Updates pending buffer state after draining.
static inline void iree_tokenizer_replace_multi_update_pending(
    iree_tokenizer_replace_multi_state_t* state, uint8_t drained) {
  state->pending_offset += drained;
  state->pending_length -= drained;
  if (state->pending_length == 0) {
    state->pending_offset = 0;
  }
}

// Sets pending buffer with content that couldn't be emitted.
static inline void iree_tokenizer_replace_multi_set_pending(
    iree_tokenizer_replace_multi_state_t* state, const uint8_t* data,
    uint8_t offset, uint8_t length) {
  memcpy(state->pending_buffer, data + offset, length);
  state->pending_offset = 0;
  state->pending_length = length;
}

// Drains pending buffer to output.
// Returns true if fully drained, false if output exhausted.
static inline bool iree_tokenizer_replace_multi_drain_pending(
    iree_tokenizer_replace_multi_state_t* IREE_RESTRICT state,
    uint8_t** out_ptr, uint8_t* out_end) {
  if (state->pending_length == 0) {
    return true;
  }

  iree_host_size_t output_available = (iree_host_size_t)(out_end - *out_ptr);
  iree_host_size_t to_write = state->pending_length < output_available
                                  ? state->pending_length
                                  : output_available;
  if (to_write > 0) {
    memcpy(*out_ptr, state->pending_buffer + state->pending_offset, to_write);
    *out_ptr += to_write;
    iree_tokenizer_replace_multi_update_pending(state, (uint8_t)to_write);
  }
  return state->pending_length == 0;
}

// Emits content to output, using pending buffer if output is too small.
// Returns number of bytes written to output.
static inline iree_host_size_t iree_tokenizer_replace_multi_emit_content(
    iree_tokenizer_replace_multi_state_t* IREE_RESTRICT state,
    uint8_t** out_ptr, uint8_t* out_end) {
  if (state->content_length == 0) {
    return 0;  // Deletion.
  }

  iree_host_size_t output_available = (iree_host_size_t)(out_end - *out_ptr);
  if (output_available >= state->content_length) {
    // Fast path: full content fits.
    memcpy(*out_ptr, state->content, state->content_length);
    *out_ptr += state->content_length;
    return state->content_length;
  }

  // Slow path: partial write, buffer remainder.
  if (output_available > 0) {
    memcpy(*out_ptr, state->content, output_available);
    *out_ptr += output_available;
  }
  iree_host_size_t remaining = state->content_length - output_available;
  iree_tokenizer_replace_multi_set_pending(
      state, state->content, (uint8_t)output_available, (uint8_t)remaining);
  return output_available;
}

// Emits a single byte to output.
static inline void iree_tokenizer_replace_multi_emit_byte(
    iree_tokenizer_replace_multi_state_t* IREE_RESTRICT state, uint8_t byte,
    uint8_t** out_ptr, uint8_t* out_end) {
  if (*out_ptr < out_end) {
    *(*out_ptr)++ = byte;
  } else {
    state->pending_buffer[0] = byte;
    state->pending_offset = 0;
    state->pending_length = 1;
  }
}

static iree_status_t iree_tokenizer_replace_multi_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_normalizer_replace_t* self =
      (const iree_tokenizer_normalizer_replace_t*)normalizer;
  iree_tokenizer_replace_multi_state_t* state =
      (iree_tokenizer_replace_multi_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  state->pattern = iree_tokenizer_normalizer_replace_pattern(self);
  state->content = iree_tokenizer_normalizer_replace_content(self);
  state->pattern_length = self->pattern_length;
  state->content_length = self->content_length;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_replace_multi_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_replace_multi_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  (void)flags;  // Replace has no cross-segment state to reset.

  iree_tokenizer_replace_multi_state_t* state =
      (iree_tokenizer_replace_multi_state_t*)base_state;

  const uint8_t* in_ptr = (const uint8_t*)input.data;
  const uint8_t* in_end = in_ptr + input.size;
  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  const uint8_t* pattern = state->pattern;
  const uint8_t pattern_length = state->pattern_length;
  const uint8_t first_byte = pattern[0];

  // Drain any pending content from previous call.
  if (!iree_tokenizer_replace_multi_drain_pending(state, &out_ptr, out_end)) {
    *out_consumed = 0;
    *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
    return iree_ok_status();
  }

  // Phase 1: Handle cross-boundary matches using overlap buffer.
  // The overlap buffer holds the last (pattern_length - 1) bytes from the
  // previous chunk that might be the start of a pattern spanning chunks.
  if (state->overlap_length > 0) {
    iree_host_size_t total_available = state->overlap_length + input.size;
    iree_host_size_t max_overlap = pattern_length - 1;

    if (total_available < pattern_length) {
      // Not enough data yet to check for a match. Append input to overlap.
      // If this would overflow, emit oldest bytes first.
      while (state->overlap_length + input.size > max_overlap &&
             out_ptr < out_end && state->pending_length == 0) {
        iree_tokenizer_replace_multi_emit_byte(state, state->overlap_buffer[0],
                                               &out_ptr, out_end);
        memmove(state->overlap_buffer, state->overlap_buffer + 1,
                state->overlap_length - 1);
        state->overlap_length--;
      }
      // Append remaining input to overlap.
      if (state->overlap_length + input.size <= max_overlap) {
        memcpy(state->overlap_buffer + state->overlap_length, in_ptr,
               input.size);
        state->overlap_length += (uint8_t)input.size;
        *out_consumed = input.size;
        *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
        return iree_ok_status();
      } else {
        // Output buffer full before we could make room. Return to let caller
        // flush output and retry with the same input.
        *out_consumed = 0;
        *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
        return iree_ok_status();
      }
    }

    // We have enough data to check for cross-boundary matches.
    // Build a temporary buffer: [overlap] + [enough input bytes to check]
    uint8_t cross_buffer[2 * IREE_TOKENIZER_REPLACE_MAX_PATTERN];
    iree_host_size_t cross_length = state->overlap_length;
    memcpy(cross_buffer, state->overlap_buffer, cross_length);

    // Add bytes from input. We need pattern_length bytes to fully check
    // patterns that could start at any position in the overlap. A pattern
    // starting at the LAST overlap position needs all pattern_length bytes
    // from input to complete.
    iree_host_size_t bytes_to_add = pattern_length;
    if (bytes_to_add > input.size) bytes_to_add = input.size;
    memcpy(cross_buffer + cross_length, in_ptr, bytes_to_add);
    cross_length += bytes_to_add;

    // Scan cross buffer for matches, processing positions from overlap.
    // We can only check positions where we have pattern_length bytes available.
    iree_host_size_t i = 0;
    iree_host_size_t original_overlap_length = state->overlap_length;
    while (i + pattern_length <= cross_length && out_ptr < out_end &&
           state->pending_length == 0) {
      if (cross_buffer[i] == first_byte &&
          memcmp(cross_buffer + i, pattern, pattern_length) == 0) {
        // Match found! Emit content.
        iree_tokenizer_replace_multi_emit_content(state, &out_ptr, out_end);
        i += pattern_length;
      } else {
        // No match at position i. Emit the byte.
        iree_tokenizer_replace_multi_emit_byte(state, cross_buffer[i], &out_ptr,
                                               out_end);
        ++i;
      }
    }

    // After scanning, bytes from position i to cross_length-1 haven't been
    // checked (not enough bytes for a full pattern check). These could be
    // the start of a pattern spanning to the next chunk, so save them to
    // the overlap buffer.
    iree_host_size_t unchecked_length = cross_length - i;
    if (unchecked_length > max_overlap) {
      // Can't fit all unchecked bytes. Emit the oldest ones that can't
      // possibly start a match (would need more than max_overlap bytes).
      iree_host_size_t to_emit = unchecked_length - max_overlap;
      for (iree_host_size_t j = 0;
           j < to_emit && out_ptr < out_end && state->pending_length == 0;
           ++j) {
        iree_tokenizer_replace_multi_emit_byte(state, cross_buffer[i + j],
                                               &out_ptr, out_end);
      }
      i += to_emit;
      unchecked_length = max_overlap;
    }

    // Save unchecked suffix to overlap buffer for next chunk.
    if (unchecked_length > 0) {
      memcpy(state->overlap_buffer, cross_buffer + i, unchecked_length);
    }
    state->overlap_length = (uint8_t)unchecked_length;

    // Calculate how many input bytes were consumed.
    // Input bytes start at position original_overlap_length in cross_buffer.
    // We're keeping bytes from position i onwards in overlap.
    // So input bytes consumed = min(i, original_overlap_length) input bytes
    // that were checked/emitted, plus any that are now in the new overlap.
    iree_host_size_t total_processed = i + unchecked_length;
    iree_host_size_t input_bytes_in_cross = bytes_to_add;
    if (total_processed > original_overlap_length) {
      // We processed beyond the original overlap, consuming some input.
      iree_host_size_t input_consumed_from_cross =
          total_processed - original_overlap_length;
      if (input_consumed_from_cross > input_bytes_in_cross) {
        input_consumed_from_cross = input_bytes_in_cross;
      }
      in_ptr += input_consumed_from_cross;
    }

    // Check if we ran out of output space.
    if (out_ptr >= out_end || state->pending_length > 0) {
      *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Phase 2: Main chunk scan with memchr + memcmp.
  // Scan up to the "safe region" where we're guaranteed to have pattern_length
  // bytes available for comparison.
  iree_host_size_t remaining_input = (iree_host_size_t)(in_end - in_ptr);
  if (remaining_input >= pattern_length) {
    const uint8_t* safe_end = in_end - pattern_length + 1;
    const uint8_t* scan = in_ptr;

    while (scan < safe_end && out_ptr < out_end && state->pending_length == 0) {
      // SIMD-accelerated search for first byte of pattern.
      const uint8_t* found =
          (const uint8_t*)memchr(scan, first_byte, (size_t)(safe_end - scan));
      if (!found) {
        // No more first bytes in safe region. Emit all scanned bytes.
        iree_host_size_t emit_length = (iree_host_size_t)(safe_end - scan);
        iree_host_size_t output_available =
            (iree_host_size_t)(out_end - out_ptr);
        if (output_available >= emit_length) {
          memcpy(out_ptr, scan, emit_length);
          out_ptr += emit_length;
          scan = safe_end;
        } else {
          memcpy(out_ptr, scan, output_available);
          out_ptr += output_available;
          scan += output_available;
        }
        break;
      }

      // Emit bytes before the potential match.
      if (found > scan) {
        iree_host_size_t emit_length = (iree_host_size_t)(found - scan);
        iree_host_size_t output_available =
            (iree_host_size_t)(out_end - out_ptr);
        if (output_available >= emit_length) {
          memcpy(out_ptr, scan, emit_length);
          out_ptr += emit_length;
        } else {
          memcpy(out_ptr, scan, output_available);
          out_ptr += output_available;
          scan += output_available;
          break;  // Output full.
        }
      }

      // Check for full pattern match.
      if (memcmp(found, pattern, pattern_length) == 0) {
        // Match! Emit content.
        iree_tokenizer_replace_multi_emit_content(state, &out_ptr, out_end);
        scan = found + pattern_length;
      } else {
        // Mismatch. Emit the first byte and continue.
        iree_tokenizer_replace_multi_emit_byte(state, *found, &out_ptr,
                                               out_end);
        scan = found + 1;
      }
    }

    in_ptr = scan;
  }

  // Emit UTF-8 continuation bytes that trail the Phase 2 safe boundary.
  // Valid UTF-8 patterns always start with an ASCII byte (0x00-0x7F) or a
  // lead byte (0xC2-0xF4), never a continuation byte (0x80-0xBF). Any
  // continuation bytes at the start of the remaining tail belong to a character
  // whose lead byte was already emitted by Phase 2. Emitting them ensures the
  // output ends on a UTF-8 character boundary, which is required by the
  // normalizer interface contract (downstream normalizers in a sequence expect
  // complete codepoints).
  while (in_ptr < in_end && (*in_ptr & 0xC0) == 0x80 && out_ptr < out_end &&
         state->pending_length == 0) {
    *out_ptr++ = *in_ptr++;
  }

  // Phase 3: Buffer tail bytes for next chunk.
  // The last (pattern_length - 1) bytes may be the start of a cross-boundary
  // match, so append them to the overlap buffer. Note: Phase 1 may have
  // already placed bytes in the overlap buffer, so we APPEND rather than
  // overwrite.
  iree_host_size_t tail_length = (iree_host_size_t)(in_end - in_ptr);
  if (tail_length > 0) {
    iree_host_size_t max_overlap = pattern_length - 1;
    iree_host_size_t current_overlap = state->overlap_length;
    iree_host_size_t available_space = max_overlap - current_overlap;

    if (tail_length <= available_space) {
      // All remaining bytes fit in overlap buffer.
      memcpy(state->overlap_buffer + current_overlap, in_ptr, tail_length);
      state->overlap_length = (uint8_t)(current_overlap + tail_length);
      in_ptr = in_end;
    } else {
      // Not all bytes fit - emit the excess, keep max_overlap bytes total.
      // First, emit bytes that definitely can't be part of a cross-boundary
      // match (too far from end to start a pattern).
      iree_host_size_t total_needed = current_overlap + tail_length;
      iree_host_size_t excess = total_needed - max_overlap;

      // Emit from oldest data first (current overlap buffer).
      iree_host_size_t emit_from_overlap =
          excess < current_overlap ? excess : current_overlap;
      for (iree_host_size_t j = 0; j < emit_from_overlap && out_ptr < out_end &&
                                   state->pending_length == 0;
           ++j) {
        iree_tokenizer_replace_multi_emit_byte(state, state->overlap_buffer[j],
                                               &out_ptr, out_end);
      }
      if (emit_from_overlap > 0 && emit_from_overlap < current_overlap) {
        memmove(state->overlap_buffer,
                state->overlap_buffer + emit_from_overlap,
                current_overlap - emit_from_overlap);
      }
      current_overlap -= emit_from_overlap;
      excess -= emit_from_overlap;

      // Then emit from input if still have excess.
      iree_host_size_t emit_from_input = excess;
      iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
      iree_host_size_t to_emit = emit_from_input < output_available
                                     ? emit_from_input
                                     : output_available;
      if (to_emit > 0 && state->pending_length == 0) {
        memcpy(out_ptr, in_ptr, to_emit);
        out_ptr += to_emit;
        in_ptr += to_emit;
      }

      // Append remaining input to overlap buffer.
      tail_length = (iree_host_size_t)(in_end - in_ptr);
      available_space = max_overlap - current_overlap;
      if (tail_length > 0 && tail_length <= available_space) {
        memcpy(state->overlap_buffer + current_overlap, in_ptr, tail_length);
        state->overlap_length = (uint8_t)(current_overlap + tail_length);
        in_ptr = in_end;
      }
    }
  }

  *out_consumed = (iree_host_size_t)(in_ptr - (const uint8_t*)input.data);
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_replace_multi_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_replace_multi_state_t* state =
      (iree_tokenizer_replace_multi_state_t*)base_state;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Drain pending buffer first.
  if (!iree_tokenizer_replace_multi_drain_pending(state, &out_ptr, out_end)) {
    *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
    return iree_ok_status();
  }

  // Flush overlap buffer - these bytes didn't form a complete match.
  if (state->overlap_length > 0) {
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    iree_host_size_t to_write = state->overlap_length < output_available
                                    ? state->overlap_length
                                    : output_available;
    if (to_write > 0) {
      memcpy(out_ptr, state->overlap_buffer, to_write);
      out_ptr += to_write;
      // Shift remaining bytes in overlap buffer.
      state->overlap_length -= (uint8_t)to_write;
      if (state->overlap_length > 0) {
        memmove(state->overlap_buffer, state->overlap_buffer + to_write,
                state->overlap_length);
      }
    }
  }

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static bool iree_tokenizer_replace_multi_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_replace_multi_state_t* state =
      (const iree_tokenizer_replace_multi_state_t*)base_state;
  // Returns true only if there's content that couldn't fit in output
  // (pending_buffer). The overlap_buffer holds data waiting for more INPUT
  // to complete pattern matching, not waiting for more OUTPUT space.
  // Callers should continue providing input or call finalize() to flush.
  return state->pending_length > 0;
}

//===----------------------------------------------------------------------===//
// Vtables
//===----------------------------------------------------------------------===//

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_replace_single_vtable = {
        .destroy = iree_tokenizer_normalizer_replace_destroy,
        .state_initialize = iree_tokenizer_replace_single_state_initialize,
        .state_deinitialize = iree_tokenizer_replace_single_state_deinitialize,
        .state_process = iree_tokenizer_replace_single_state_process,
        .state_finalize = iree_tokenizer_replace_single_state_finalize,
        .state_has_pending = iree_tokenizer_replace_single_state_has_pending,
};

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_replace_multi_vtable = {
        .destroy = iree_tokenizer_normalizer_replace_destroy,
        .state_initialize = iree_tokenizer_replace_multi_state_initialize,
        .state_deinitialize = iree_tokenizer_replace_multi_state_deinitialize,
        .state_process = iree_tokenizer_replace_multi_state_process,
        .state_finalize = iree_tokenizer_replace_multi_state_finalize,
        .state_has_pending = iree_tokenizer_replace_multi_state_has_pending,
};
