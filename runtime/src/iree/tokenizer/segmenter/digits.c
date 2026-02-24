// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/digits.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Digits Segmenter Implementation
//===----------------------------------------------------------------------===//

static inline bool iree_is_ascii_digit(uint8_t byte) {
  return byte >= '0' && byte <= '9';
}

typedef struct {
  iree_tokenizer_segment_output_t output;
  // Absolute offset of this chunk's start.
  iree_host_size_t chunk_base;
  iree_host_size_t count;  // Segments emitted.
  // Absolute end of last emitted segment.
  iree_host_size_t last_end;
  // True if output capacity exhausted.
  bool full;
} iree_tokenizer_digits_emitter_t;

// Emits a segment [start, end) where start/end are ABSOLUTE byte offsets.
// Converts to relative offsets for the output. Returns true if emitted.
static inline bool iree_tokenizer_digits_emit(
    iree_tokenizer_digits_emitter_t* emitter, iree_host_size_t start,
    iree_host_size_t end) {
  if (start >= end || emitter->full) return false;
  if (emitter->count >= emitter->output.capacity) {
    emitter->full = true;
    return false;
  }
  emitter->output.values[emitter->count].start = start - emitter->chunk_base;
  emitter->output.values[emitter->count].end = end - emitter->chunk_base;
  emitter->count++;
  emitter->last_end = end;
  return true;
}

typedef struct iree_tokenizer_segmenter_digits_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
  bool individual_digits;
} iree_tokenizer_segmenter_digits_t;

typedef struct iree_tokenizer_segmenter_digits_state_t {
  iree_tokenizer_segmenter_state_t base;

  // Cumulative bytes consumed across all process() calls.
  iree_host_size_t bytes_consumed;
  // Absolute end of last emitted segment (for gap tracking).
  iree_host_size_t last_emit_end;

  // Active segment tracking (for has_pending).
  // True if we're mid-segment at end of process().
  bool in_segment;
  // Type of segment being accumulated.
  bool segment_is_digits;

  // Process overflow: segment found but output buffer was full.
  bool process_has_pending;
  iree_host_size_t process_pending_start;
  iree_host_size_t process_pending_end;

  // Finalize overflow: finalize found segment but output buffer was full.
  bool finalize_has_pending;
  iree_host_size_t finalize_pending_start;
  iree_host_size_t finalize_pending_end;
  iree_host_size_t finalize_scan_position;
} iree_tokenizer_segmenter_digits_state_t;

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_digits_vtable;

iree_status_t iree_tokenizer_segmenter_digits_allocate(
    bool individual_digits, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  iree_tokenizer_segmenter_digits_t* segmenter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter));

  iree_tokenizer_segmenter_initialize(
      &segmenter->base, &iree_tokenizer_segmenter_digits_vtable,
      sizeof(iree_tokenizer_segmenter_digits_state_t));
  segmenter->allocator = allocator;
  segmenter->individual_digits = individual_digits;

  *out_segmenter = &segmenter->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_digits_destroy(
    iree_tokenizer_segmenter_t* segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_digits_t* self =
      (iree_tokenizer_segmenter_digits_t*)segmenter;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_digits_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_digits_state_t* state =
      (iree_tokenizer_segmenter_digits_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.segmenter = segmenter;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_digits_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Core Scanner (shared between process and finalize)
//===----------------------------------------------------------------------===//

// Scans input for digit boundaries and emits segments.
// Returns the scan position where we stopped (may be < input.size on overflow).
// Updates emitter with emitted segments.
static iree_host_size_t iree_tokenizer_digits_scan(
    const char* data, iree_host_size_t size, iree_host_size_t chunk_base,
    iree_host_size_t scan_start, bool individual_digits,
    iree_tokenizer_digits_emitter_t* emitter,
    iree_host_size_t* inout_segment_start, bool* inout_segment_is_digits) {
  iree_host_size_t position = scan_start;
  iree_host_size_t segment_start = *inout_segment_start;
  bool segment_is_digits = *inout_segment_is_digits;
  bool in_segment = (segment_start < chunk_base + position);

  while (position < size) {
    uint8_t byte = (uint8_t)data[position];
    bool is_digit = iree_is_ascii_digit(byte);
    iree_host_size_t abs_position = chunk_base + position;

    if (!in_segment) {
      // Start a new segment.
      in_segment = true;
      segment_start = abs_position;
      segment_is_digits = is_digit;
      position++;

      // If individual_digits and this is a digit, emit immediately.
      if (individual_digits && is_digit) {
        if (!iree_tokenizer_digits_emit(emitter, segment_start,
                                        abs_position + 1)) {
          // Overflow. Back up so caller can retry.
          *inout_segment_start = segment_start;
          *inout_segment_is_digits = is_digit;
          return position - 1;
        }
        in_segment = false;
        segment_start = abs_position + 1;  // Past the emitted segment.
      }
    } else {
      // Check if we need to end current segment.
      bool should_end = false;
      if (segment_is_digits) {
        if (!is_digit || individual_digits) {
          should_end = true;
        }
      } else {
        if (is_digit) {
          should_end = true;
        }
      }

      if (should_end) {
        // Emit [segment_start, abs_position).
        if (!iree_tokenizer_digits_emit(emitter, segment_start, abs_position)) {
          // Overflow. Return position of segment start for retry.
          *inout_segment_start = segment_start;
          *inout_segment_is_digits = segment_is_digits;
          return segment_start - chunk_base;
        }

        // Start new segment.
        segment_start = abs_position;
        segment_is_digits = is_digit;
        position++;

        // If individual_digits and this is a digit, emit immediately.
        if (individual_digits && is_digit) {
          if (!iree_tokenizer_digits_emit(emitter, segment_start,
                                          abs_position + 1)) {
            *inout_segment_start = segment_start;
            *inout_segment_is_digits = is_digit;
            return position - 1;
          }
          in_segment = false;
          segment_start = abs_position + 1;  // Past the emitted segment.
        }
      } else {
        position++;
      }
    }
  }

  // Update state for caller.
  *inout_segment_start = segment_start;
  *inout_segment_is_digits = segment_is_digits;
  return position;
}

//===----------------------------------------------------------------------===//
// Process
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_segmenter_digits_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_digits_state_t* self =
      (iree_tokenizer_segmenter_digits_state_t*)state;
  const iree_tokenizer_segmenter_digits_t* segmenter =
      (const iree_tokenizer_segmenter_digits_t*)state->segmenter;

  *out_consumed = 0;
  *out_segment_count = 0;

  if (input.size == 0) {
    return iree_ok_status();
  }

  iree_host_size_t chunk_base = self->bytes_consumed;

  iree_tokenizer_digits_emitter_t emitter = {
      .output = output,
      .chunk_base = chunk_base,
      .count = 0,
      .last_end = self->last_emit_end,
      .full = false,
  };

  // Emit pending segment from prior call if any.
  if (self->process_has_pending) {
    if (!iree_tokenizer_digits_emit(&emitter, self->process_pending_start,
                                    self->process_pending_end)) {
      *out_segment_count = 0;
      return iree_ok_status();
    }
    self->process_has_pending = false;
  }

  // Scan input. Use saved segment state if we were mid-segment.
  iree_host_size_t segment_start =
      self->in_segment ? self->last_emit_end : chunk_base;
  bool segment_is_digits = self->segment_is_digits;
  if (!self->in_segment && input.size > 0) {
    segment_is_digits = iree_is_ascii_digit((uint8_t)input.data[0]);
  }

  iree_host_size_t scan_end = iree_tokenizer_digits_scan(
      input.data, input.size, chunk_base, 0, segmenter->individual_digits,
      &emitter, &segment_start, &segment_is_digits);

  // Update state.
  self->last_emit_end = emitter.last_end;

  // Determine consumption and in_segment state.
  // If we're mid-segment at end of input, don't consume those bytes.
  if (emitter.full) {
    // Overflow: consume up to last emitted segment.
    if (emitter.last_end > chunk_base) {
      *out_consumed = emitter.last_end - chunk_base;
      self->bytes_consumed = emitter.last_end;
    } else {
      *out_consumed = 0;
    }
    // Mid-segment because we couldn't emit everything.
    self->in_segment = true;
    self->segment_is_digits = segment_is_digits;
  } else if (segment_start < chunk_base + scan_end) {
    // Mid-segment: a segment extends to end of input without being emitted.
    // Don't consume those bytes - leave them for finalize.
    iree_host_size_t relative_start = segment_start - chunk_base;
    *out_consumed = relative_start;
    self->bytes_consumed = segment_start;
    self->in_segment = true;
    self->segment_is_digits = segment_is_digits;
  } else {
    // All segments complete.
    *out_consumed = scan_end;
    self->bytes_consumed = chunk_base + scan_end;
    self->in_segment = false;
  }

  *out_segment_count = emitter.count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Finalize
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_segmenter_digits_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t remaining_input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_digits_state_t* self =
      (iree_tokenizer_segmenter_digits_state_t*)state;
  const iree_tokenizer_segmenter_digits_t* segmenter =
      (const iree_tokenizer_segmenter_digits_t*)state->segmenter;

  *out_segment_count = 0;

  iree_host_size_t chunk_base = self->bytes_consumed;

  iree_tokenizer_digits_emitter_t emitter = {
      .output = output,
      .chunk_base = chunk_base,
      .count = 0,
      .last_end = self->last_emit_end,
      .full = false,
  };

  // Emit pending segment from prior finalize call if any.
  if (self->finalize_has_pending) {
    if (!iree_tokenizer_digits_emit(&emitter, self->finalize_pending_start,
                                    self->finalize_pending_end)) {
      *out_segment_count = 0;
      return iree_ok_status();
    }
    self->finalize_has_pending = false;
  }

  if (remaining_input.size == 0) {
    // Nothing to finalize - clear in_segment.
    self->in_segment = false;
    *out_segment_count = emitter.count;
    return iree_ok_status();
  }

  // Scan remaining input.
  iree_host_size_t segment_start = chunk_base + self->finalize_scan_position;
  bool segment_is_digits = self->segment_is_digits;
  if (self->finalize_scan_position < remaining_input.size) {
    segment_is_digits = iree_is_ascii_digit(
        (uint8_t)remaining_input.data[self->finalize_scan_position]);
  }

  iree_host_size_t scan_end = iree_tokenizer_digits_scan(
      remaining_input.data, remaining_input.size, chunk_base,
      self->finalize_scan_position, segmenter->individual_digits, &emitter,
      &segment_start, &segment_is_digits);

  // Emit final segment if any remaining.
  if (scan_end == remaining_input.size &&
      segment_start < chunk_base + scan_end) {
    if (!iree_tokenizer_digits_emit(&emitter, segment_start,
                                    chunk_base + scan_end)) {
      // Save for next call.
      self->finalize_has_pending = true;
      self->finalize_pending_start = segment_start;
      self->finalize_pending_end = chunk_base + scan_end;
      self->finalize_scan_position = scan_end;
      *out_segment_count = emitter.count;
      return iree_ok_status();
    }
  }

  if (emitter.full) {
    // Overflow: save position for next call.
    self->finalize_scan_position = emitter.last_end - chunk_base;
    if (segment_start > emitter.last_end) {
      self->finalize_has_pending = true;
      self->finalize_pending_start = segment_start;
      self->finalize_pending_end = chunk_base + scan_end;
    }
  } else {
    // All done - clear state.
    self->in_segment = false;
    self->finalize_scan_position = 0;
  }

  *out_segment_count = emitter.count;
  return iree_ok_status();
}

static bool iree_tokenizer_segmenter_digits_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  const iree_tokenizer_segmenter_digits_state_t* self =
      (const iree_tokenizer_segmenter_digits_state_t*)state;
  // Three pending conditions:
  // - process_has_pending: segment found during process() but output was full
  // - in_segment: mid-segment at end of process(), needs finalize to emit
  // - finalize_has_pending: segment found during finalize() but output was full
  return self->process_has_pending || self->in_segment ||
         self->finalize_has_pending;
}

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_digits_vtable = {
        .destroy = iree_tokenizer_segmenter_digits_destroy,
        .state_initialize = iree_tokenizer_segmenter_digits_state_initialize,
        .state_deinitialize =
            iree_tokenizer_segmenter_digits_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_digits_state_process,
        .state_finalize = iree_tokenizer_segmenter_digits_state_finalize,
        .state_has_pending = iree_tokenizer_segmenter_digits_state_has_pending,
};
