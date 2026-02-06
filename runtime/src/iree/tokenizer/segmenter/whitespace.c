// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/whitespace.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Whitespace Segmenter Implementation
//===----------------------------------------------------------------------===//

// Whitespace segmenter: no additional config beyond base.
typedef struct iree_tokenizer_segmenter_whitespace_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
} iree_tokenizer_segmenter_whitespace_t;

// Whitespace state: tracks if we're inside a segment.
typedef struct iree_tokenizer_segmenter_whitespace_state_t {
  iree_tokenizer_segmenter_state_t base;
  // True if we've started a segment but haven't found its end yet.
  bool in_segment;
  // Start position of current segment (cumulative offset across all calls).
  // When in_segment is true, this marks where the segment began.
  iree_host_size_t segment_start;
  // Total bytes processed across all process() calls.
  // Used to compute cumulative segment positions and final segment end.
  iree_host_size_t bytes_processed;

  // Finalize state: tracks pending segment and scan position for incremental
  // finalize with limited output capacity.
  // True if we have a segment to emit.
  bool finalize_has_pending;
  iree_host_size_t finalize_pending_start;  // Start of pending segment.
  iree_host_size_t finalize_pending_end;    // End of pending segment.
  iree_host_size_t finalize_scan_position;  // Where to resume scanning.
} iree_tokenizer_segmenter_whitespace_state_t;

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_whitespace_vtable;

iree_status_t iree_tokenizer_segmenter_whitespace_allocate(
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  iree_tokenizer_segmenter_whitespace_t* segmenter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter));

  iree_tokenizer_segmenter_initialize(
      &segmenter->base, &iree_tokenizer_segmenter_whitespace_vtable,
      sizeof(iree_tokenizer_segmenter_whitespace_state_t));
  segmenter->allocator = allocator;

  *out_segmenter = &segmenter->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_whitespace_destroy(
    iree_tokenizer_segmenter_t* segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_whitespace_t* self =
      (iree_tokenizer_segmenter_whitespace_t*)segmenter;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_whitespace_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_whitespace_state_t* state =
      (iree_tokenizer_segmenter_whitespace_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.segmenter = segmenter;
  state->in_segment = false;
  state->segment_start = 0;
  state->bytes_processed = 0;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_whitespace_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Returns true if the byte is ASCII whitespace.
static inline bool iree_is_ascii_whitespace(uint8_t byte) {
  // Space, tab, newline, carriage return, form feed, vertical tab.
  return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' ||
         byte == '\f' || byte == '\v';
}

static iree_status_t iree_tokenizer_segmenter_whitespace_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_whitespace_state_t* self =
      (iree_tokenizer_segmenter_whitespace_state_t*)state;

  *out_consumed = 0;
  *out_segment_count = 0;

  if (input.size == 0) {
    return iree_ok_status();
  }

  iree_host_size_t segment_count = 0;
  iree_host_size_t position = 0;

  // Note: segment_start is NOT updated here since it was set to absolute
  // position when the segment started. It remains valid across calls.

  // Process input byte by byte.
  while (position < input.size) {
    uint8_t byte = (uint8_t)input.data[position];
    bool is_whitespace = iree_is_ascii_whitespace(byte);

    if (!self->in_segment) {
      // Looking for segment start.
      if (!is_whitespace) {
        // Start of a new segment. Store absolute position.
        self->in_segment = true;
        self->segment_start = self->bytes_processed + position;
      }
      // Either way, consume this byte.
      position++;
    } else {
      // Inside a segment, looking for end.
      if (is_whitespace) {
        // Found end of segment. Emit it if we have capacity.
        if (segment_count >= output.capacity) {
          // No room for this segment. Return consumed up to segment start
          // relative to this input's start.
          iree_host_size_t relative_start =
              self->segment_start - self->bytes_processed;
          self->in_segment = false;  // Reset so next call starts fresh.
          *out_consumed = relative_start;
          self->bytes_processed += relative_start;
          *out_segment_count = segment_count;
          return iree_ok_status();
        }
        // Emit segment with positions relative to input start (for tokenizer
        // adjustment).
        output.values[segment_count].start =
            self->segment_start - self->bytes_processed;
        output.values[segment_count].end = position;
        segment_count++;
        self->in_segment = false;
        position++;  // Consume the whitespace.
      } else {
        // Still in segment, continue.
        position++;
      }
    }
  }

  // Pull-based: if we're mid-segment at end of input, don't consume those
  // bytes. Leave them for finalize() which will receive them as
  // remaining_input.
  if (self->in_segment) {
    // segment_start is absolute; convert to relative for this input chunk.
    iree_host_size_t relative_start =
        self->segment_start - self->bytes_processed;
    self->bytes_processed += relative_start;
    *out_consumed = relative_start;
  } else {
    self->bytes_processed += position;
    *out_consumed = position;
  }
  *out_segment_count = segment_count;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_segmenter_whitespace_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t remaining_input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_whitespace_state_t* self =
      (iree_tokenizer_segmenter_whitespace_state_t*)state;

  iree_host_size_t segment_count = 0;

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
    self->finalize_has_pending = false;

    // If that filled output, we're done for this call.
    if (segment_count >= output.capacity) {
      *out_segment_count = segment_count;
      return iree_ok_status();
    }
  }

  // Resume scanning from saved position (0 on first call).
  iree_host_size_t position = self->finalize_scan_position;

  // Extract whitespace-delimited segments from remaining_input.
  while (position < remaining_input.size) {
    // Skip any leading whitespace.
    while (position < remaining_input.size &&
           iree_is_ascii_whitespace((uint8_t)remaining_input.data[position])) {
      ++position;
    }

    if (position >= remaining_input.size) {
      break;  // Only trailing whitespace.
    }

    // Found segment start.
    iree_host_size_t start = position;

    // Find segment end (next whitespace or end of input).
    while (position < remaining_input.size &&
           !iree_is_ascii_whitespace((uint8_t)remaining_input.data[position])) {
      ++position;
    }

    // Emit segment if we have capacity.
    if (segment_count >= output.capacity) {
      // No room - save this segment for next call.
      self->finalize_has_pending = true;
      self->finalize_pending_start = start;
      self->finalize_pending_end = position;
      self->finalize_scan_position = position;
      self->in_segment = true;  // Signal has_pending.
      *out_segment_count = segment_count;
      return iree_ok_status();
    }

    output.values[segment_count].start = start;
    output.values[segment_count].end = position;
    ++segment_count;
  }

  // All done - clear finalize state.
  self->finalize_scan_position = 0;
  self->in_segment = false;
  *out_segment_count = segment_count;
  return iree_ok_status();
}

static bool iree_tokenizer_segmenter_whitespace_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  const iree_tokenizer_segmenter_whitespace_state_t* self =
      (const iree_tokenizer_segmenter_whitespace_state_t*)state;
  // Has pending if we're in the middle of a segment.
  return self->in_segment;
}

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_whitespace_vtable = {
        .destroy = iree_tokenizer_segmenter_whitespace_destroy,
        .state_initialize =
            iree_tokenizer_segmenter_whitespace_state_initialize,
        .state_deinitialize =
            iree_tokenizer_segmenter_whitespace_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_whitespace_state_process,
        .state_finalize = iree_tokenizer_segmenter_whitespace_state_finalize,
        .state_has_pending =
            iree_tokenizer_segmenter_whitespace_state_has_pending,
};
