// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/metaspace.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// UTF-8 Encoding Helper
//===----------------------------------------------------------------------===//

// Encodes a Unicode codepoint to UTF-8.
// Returns the number of bytes written (1-4), or 0 on invalid codepoint.
static uint8_t iree_unicode_codepoint_to_utf8(uint32_t codepoint,
                                              uint8_t* out_bytes) {
  if (codepoint <= 0x7F) {
    out_bytes[0] = (uint8_t)codepoint;
    return 1;
  } else if (codepoint <= 0x7FF) {
    out_bytes[0] = (uint8_t)(0xC0 | (codepoint >> 6));
    out_bytes[1] = (uint8_t)(0x80 | (codepoint & 0x3F));
    return 2;
  } else if (codepoint <= 0xFFFF) {
    out_bytes[0] = (uint8_t)(0xE0 | (codepoint >> 12));
    out_bytes[1] = (uint8_t)(0x80 | ((codepoint >> 6) & 0x3F));
    out_bytes[2] = (uint8_t)(0x80 | (codepoint & 0x3F));
    return 3;
  } else if (codepoint <= 0x10FFFF) {
    out_bytes[0] = (uint8_t)(0xF0 | (codepoint >> 18));
    out_bytes[1] = (uint8_t)(0x80 | ((codepoint >> 12) & 0x3F));
    out_bytes[2] = (uint8_t)(0x80 | ((codepoint >> 6) & 0x3F));
    out_bytes[3] = (uint8_t)(0x80 | (codepoint & 0x3F));
    return 4;
  }
  return 0;  // Invalid codepoint.
}

//===----------------------------------------------------------------------===//
// Metaspace Segmenter Implementation
//===----------------------------------------------------------------------===//

// Metaspace segmenter configuration.
typedef struct iree_tokenizer_segmenter_metaspace_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
  // Replacement character stored as UTF-8 bytes for direct comparison.
  uint8_t replacement_bytes[4];
  uint8_t replacement_length;
  // If false, entire input is one segment (no splitting).
  bool split_enabled;
} iree_tokenizer_segmenter_metaspace_t;

// Streaming state for Metaspace segmenter.
//
// Design: "Rewind Strategy" - no pending byte buffering.
// When a chunk ends with a potential partial delimiter (e.g., first byte of ▁),
// we DON'T consume those bytes. Instead, we return consumed = position_before,
// and the driver re-sends unconsumed bytes in the next chunk. This keeps the
// segmenter stateless regarding partial delimiter matches.
typedef struct iree_tokenizer_segmenter_metaspace_state_t {
  iree_tokenizer_segmenter_state_t base;
  // Replacement character as UTF-8 bytes (replicated for cache locality).
  uint8_t replacement_bytes[4];
  // Length of replacement in bytes (1-4).
  uint8_t replacement_length;
  // If false, entire input is one segment (no splitting).
  uint8_t split_enabled;
  // True if we're inside a segment (accumulating bytes).
  uint8_t in_segment;
  // Start position of current segment (cumulative offset across all calls).
  iree_host_size_t segment_start;
  // Total bytes processed across all process() calls.
  iree_host_size_t bytes_processed;

  // Finalize state: tracks pending segment and scan position for incremental
  // finalize with limited output capacity.
  // True if we have a segment to emit.
  uint8_t finalize_has_pending;
  iree_host_size_t finalize_pending_start;  // Start of pending segment.
  iree_host_size_t finalize_pending_end;    // End of pending segment.
  iree_host_size_t finalize_scan_position;  // Where to resume scanning.
} iree_tokenizer_segmenter_metaspace_state_t;

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_metaspace_vtable;

iree_status_t iree_tokenizer_segmenter_metaspace_allocate(
    uint32_t replacement_codepoint, bool split_enabled,
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  // Use default if 0 specified.
  if (replacement_codepoint == 0) {
    replacement_codepoint = IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT;
  }

  iree_tokenizer_segmenter_metaspace_t* segmenter = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter);

  if (iree_status_is_ok(status)) {
    iree_tokenizer_segmenter_initialize(
        &segmenter->base, &iree_tokenizer_segmenter_metaspace_vtable,
        sizeof(iree_tokenizer_segmenter_metaspace_state_t));
    segmenter->allocator = allocator;
    segmenter->split_enabled = split_enabled;

    // Encode replacement codepoint to UTF-8.
    segmenter->replacement_length = iree_unicode_codepoint_to_utf8(
        replacement_codepoint, segmenter->replacement_bytes);
    if (segmenter->replacement_length == 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid replacement codepoint U+%04X",
                                replacement_codepoint);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_segmenter = &segmenter->base;
  } else {
    iree_allocator_free(allocator, segmenter);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_tokenizer_segmenter_metaspace_destroy(
    iree_tokenizer_segmenter_t* segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_metaspace_t* self =
      (iree_tokenizer_segmenter_metaspace_t*)segmenter;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_metaspace_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_segmenter_metaspace_t* self =
      (const iree_tokenizer_segmenter_metaspace_t*)segmenter;
  iree_tokenizer_segmenter_metaspace_state_t* state =
      (iree_tokenizer_segmenter_metaspace_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.segmenter = segmenter;

  // Copy hot-path fields from segmenter to state for cache locality.
  memcpy(state->replacement_bytes, self->replacement_bytes,
         self->replacement_length);
  state->replacement_length = self->replacement_length;
  state->split_enabled = self->split_enabled ? 1 : 0;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_metaspace_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Checks if the input at the given position starts with the delimiter.
// Returns true if complete delimiter found, false otherwise.
static inline bool iree_tokenizer_segmenter_metaspace_is_delimiter(
    const iree_tokenizer_segmenter_metaspace_state_t* state,
    const uint8_t* data, iree_host_size_t remaining) {
  if (remaining < state->replacement_length) {
    return false;
  }
  return memcmp(data, state->replacement_bytes, state->replacement_length) == 0;
}

static iree_status_t iree_tokenizer_segmenter_metaspace_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_metaspace_state_t* self =
      (iree_tokenizer_segmenter_metaspace_state_t*)state;

  *out_consumed = 0;
  *out_segment_count = 0;

  if (input.size == 0) {
    return iree_ok_status();
  }

  // With zero output capacity, we can't emit segments and shouldn't consume.
  // Exception: when splitting is disabled, we never emit from process().
  if (output.capacity == 0 && self->split_enabled) {
    return iree_ok_status();
  }

  // If splitting is disabled, entire input is one segment with no boundaries.
  // Pull-based: leave all bytes for finalize() which will emit the segment.
  if (!self->split_enabled) {
    if (!self->in_segment) {
      self->in_segment = 1;
      self->segment_start = self->bytes_processed;
    }
    *out_consumed = 0;
    return iree_ok_status();
  }

  const uint8_t* data = (const uint8_t*)input.data;
  iree_host_size_t segment_count = 0;
  iree_host_size_t position = 0;

  while (position < input.size) {
    uint8_t byte = data[position];

    // Check if this could be start of delimiter.
    if (byte == self->replacement_bytes[0]) {
      iree_host_size_t remaining = input.size - position;

      if (remaining >= self->replacement_length) {
        // Have enough bytes to check for complete delimiter.
        if (iree_tokenizer_segmenter_metaspace_is_delimiter(
                self, data + position, remaining)) {
          // Complete delimiter found.
          if (self->in_segment) {
            // Emit current segment before starting new one.
            iree_host_size_t abs_end = self->bytes_processed + position;
            if (self->segment_start < abs_end) {
              if (segment_count >= output.capacity) {
                // Output full - can't emit this segment.
                // Return consumed up to segment_start (the pending segment's
                // start), not position. This leaves the pending segment's bytes
                // for the next call, ensuring they're not lost.
                //
                // Reset in_segment so the next call starts fresh. The pending
                // segment's bytes will be included in the sliced input that
                // the caller provides on the next call.
                iree_host_size_t relative_start =
                    self->segment_start - self->bytes_processed;
                self->bytes_processed = self->segment_start;
                self->in_segment = 0;
                *out_consumed = relative_start;
                *out_segment_count = segment_count;
                return iree_ok_status();
              }
              output.values[segment_count].start =
                  self->segment_start - self->bytes_processed;
              output.values[segment_count].end = position;
              segment_count++;
            }
          }
          // Start new segment at delimiter position.
          self->segment_start = self->bytes_processed + position;
          self->in_segment = 1;
          position += self->replacement_length;
          continue;
        }
      } else {
        // Potential partial delimiter at chunk boundary.
        // Can't determine if it's our delimiter or a false positive (another
        // UTF-8 character starting with 0xE2). Break and use standard end-of-
        // input handling, which will rewind to segment_start if needed.
        break;
      }
    }

    // Regular content byte.
    if (!self->in_segment) {
      self->in_segment = 1;
      self->segment_start = self->bytes_processed + position;
    }
    position++;
  }

  // End of input. Pull-based: don't consume incomplete segment bytes.
  // Leave them for finalize() which will receive them as remaining_input.
  if (self->in_segment) {
    // segment_start is absolute; compute relative position in this chunk.
    // Only consume bytes BEFORE the segment started.
    if (self->segment_start >= self->bytes_processed) {
      iree_host_size_t relative_start =
          self->segment_start - self->bytes_processed;
      self->bytes_processed += relative_start;
      *out_consumed = relative_start;
    } else {
      // Segment started in a previous chunk - we've been scanning content.
      // Consume nothing from this chunk; finalize gets entire remaining input.
      *out_consumed = 0;
    }
  } else {
    // Not in a segment - consumed everything (trailing delimiters, etc.).
    self->bytes_processed += position;
    *out_consumed = position;
  }
  *out_segment_count = segment_count;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_segmenter_metaspace_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t remaining_input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_metaspace_state_t* self =
      (iree_tokenizer_segmenter_metaspace_state_t*)state;

  iree_host_size_t segment_count = 0;

  // If splitting is disabled, emit entire remaining_input as one segment.
  if (!self->split_enabled) {
    if (remaining_input.size > 0) {
      if (output.capacity == 0) {
        self->in_segment = 1;
        *out_segment_count = 0;
        return iree_ok_status();
      }
      output.values[0].start = 0;
      output.values[0].end = remaining_input.size;
      segment_count = 1;
    }
    self->in_segment = 0;
    *out_segment_count = segment_count;
    return iree_ok_status();
  }

  // If we have a pending segment from a previous capacity-limited call,
  // emit it first.
  if (self->finalize_has_pending) {
    if (output.capacity == 0) {
      *out_segment_count = 0;
      return iree_ok_status();
    }
    output.values[segment_count].start = self->finalize_pending_start;
    output.values[segment_count].end = self->finalize_pending_end;
    ++segment_count;
    self->finalize_has_pending = 0;

    if (segment_count >= output.capacity) {
      *out_segment_count = segment_count;
      return iree_ok_status();
    }
  }

  // Resume scanning from saved position (0 on first call).
  const uint8_t* data = (const uint8_t*)remaining_input.data;
  iree_host_size_t position = self->finalize_scan_position;
  iree_host_size_t segment_start = position;
  bool in_segment = (position < remaining_input.size);

  while (position < remaining_input.size) {
    uint8_t byte = data[position];

    // Check if this is start of delimiter.
    if (byte == self->replacement_bytes[0]) {
      iree_host_size_t remaining = remaining_input.size - position;
      if (remaining >= self->replacement_length &&
          memcmp(data + position, self->replacement_bytes,
                 self->replacement_length) == 0) {
        // Delimiter found - emit previous segment if any.
        if (in_segment && position > segment_start) {
          if (segment_count >= output.capacity) {
            // Save this segment for next call.
            self->finalize_has_pending = 1;
            self->finalize_pending_start = segment_start;
            self->finalize_pending_end = position;
            self->finalize_scan_position = position;
            self->in_segment = 1;
            *out_segment_count = segment_count;
            return iree_ok_status();
          }
          output.values[segment_count].start = segment_start;
          output.values[segment_count].end = position;
          segment_count++;
        }
        // Start new segment at delimiter.
        segment_start = position;
        in_segment = true;
        position += self->replacement_length;
        continue;
      }
    }

    // Regular content.
    if (!in_segment) {
      segment_start = position;
      in_segment = true;
    }
    position++;
  }

  // Emit final segment if any.
  if (in_segment && position > segment_start) {
    if (segment_count >= output.capacity) {
      // Save this segment for next call.
      self->finalize_has_pending = 1;
      self->finalize_pending_start = segment_start;
      self->finalize_pending_end = position;
      self->finalize_scan_position = position;
      self->in_segment = 1;
      *out_segment_count = segment_count;
      return iree_ok_status();
    }
    output.values[segment_count].start = segment_start;
    output.values[segment_count].end = position;
    segment_count++;
  }

  // All done - clear finalize state.
  self->finalize_scan_position = 0;
  self->in_segment = 0;
  *out_segment_count = segment_count;
  return iree_ok_status();
}

static bool iree_tokenizer_segmenter_metaspace_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  const iree_tokenizer_segmenter_metaspace_state_t* self =
      (const iree_tokenizer_segmenter_metaspace_state_t*)state;
  return self->in_segment != 0;
}

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_metaspace_vtable = {
        .destroy = iree_tokenizer_segmenter_metaspace_destroy,
        .state_initialize = iree_tokenizer_segmenter_metaspace_state_initialize,
        .state_deinitialize =
            iree_tokenizer_segmenter_metaspace_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_metaspace_state_process,
        .state_finalize = iree_tokenizer_segmenter_metaspace_state_finalize,
        .state_has_pending =
            iree_tokenizer_segmenter_metaspace_state_has_pending,
};
