// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/punctuation.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Character Classification
//===----------------------------------------------------------------------===//

// Returns true if the codepoint is punctuation (same definition as BERT).
// Matches HuggingFace's is_punc(): ASCII punctuation OR Unicode P.
static inline bool iree_tokenizer_is_punctuation(uint32_t codepoint) {
  if ((codepoint >= 33 && codepoint <= 47) ||
      (codepoint >= 58 && codepoint <= 64) ||
      (codepoint >= 91 && codepoint <= 96) ||
      (codepoint >= 123 && codepoint <= 126)) {
    return true;
  }
  if (codepoint >= 0x80) {
    return iree_unicode_is_punctuation(codepoint);
  }
  return false;
}

// Decodes the next codepoint at |position| in |data[0..size)| and returns
// whether it is punctuation. On success, |*out_byte_length| is set to the
// number of bytes the codepoint occupies. Returns false with byte_length=0
// if the UTF-8 sequence extends past |size|.
static inline bool iree_tokenizer_punctuation_decode_next(
    const char* data, iree_host_size_t position, iree_host_size_t size,
    iree_host_size_t* out_byte_length, bool* out_is_incomplete) {
  uint8_t lead_byte = (uint8_t)data[position];
  iree_host_size_t sequence_length =
      iree_unicode_utf8_sequence_length(lead_byte);
  if (position + sequence_length > size) {
    *out_byte_length = 0;
    *out_is_incomplete = true;
    return false;
  }

  iree_string_view_t view = {data, size};
  iree_host_size_t decode_position = position;
  uint32_t codepoint = iree_unicode_utf8_decode(view, &decode_position);
  *out_byte_length = decode_position - position;
  *out_is_incomplete = false;

  return iree_tokenizer_is_punctuation(codepoint);
}

typedef struct {
  iree_tokenizer_segment_output_t output;
  iree_host_size_t chunk_base;
  iree_host_size_t count;
  // Absolute end of last successfully emitted segment.
  iree_host_size_t last_end;
  bool full;
} iree_tokenizer_punctuation_emitter_t;

// Emits a segment [start, end) where start/end are ABSOLUTE byte offsets.
// The check for empty segments is done on absolute values (before subtracting
// chunk_base). This is critical for MERGED_WITH_NEXT/CONTIGUOUS where pending
// matches from prior process() calls produce negative-wrapping relative offsets
// that the test utility's position+offset arithmetic correctly undoes.
static inline void iree_tokenizer_punctuation_emit(
    iree_tokenizer_punctuation_emitter_t* emitter, iree_host_size_t start,
    iree_host_size_t end) {
  if (start >= end || emitter->full) return;
  if (emitter->count >= emitter->output.capacity) {
    emitter->full = true;
    return;
  }
  // Validate that segment bounds are within the chunk to catch coordinate
  // system bugs early. If start < chunk_base, the subtraction would underflow.
  IREE_ASSERT(start >= emitter->chunk_base &&
              "segment start underflows chunk_base");
  IREE_ASSERT(end >= emitter->chunk_base &&
              "segment end underflows chunk_base");
  emitter->output.values[emitter->count].start = start - emitter->chunk_base;
  emitter->output.values[emitter->count].end = end - emitter->chunk_base;
  emitter->count++;
  emitter->last_end = end;
}

//===----------------------------------------------------------------------===//
// Behavior Handlers
//===----------------------------------------------------------------------===//

// Result of processing one punctuation match against its preceding gap.
typedef struct {
  iree_host_size_t segments[4];  // [start0, end0, start1, end1] (absolute)
  iree_host_size_t pending_start;
  iree_host_size_t pending_end;
  iree_host_size_t last_emit_end;
  // 0, 1, or 2 segments.
  uint8_t segment_count;
  bool has_pending;
} iree_tokenizer_punctuation_match_result_t;

// Dispatches to the appropriate behavior handler for a punctuation match.
// All coordinates are absolute byte offsets.
// |gap_start|/|gap_end| is the non-punctuation text before the match.
// |match_start|/|match_end| is the punctuation character.
static iree_tokenizer_punctuation_match_result_t
iree_tokenizer_punctuation_handle_match(
    iree_tokenizer_regex_split_behavior_t behavior, iree_host_size_t gap_start,
    iree_host_size_t gap_end, iree_host_size_t match_start,
    iree_host_size_t match_end, iree_host_size_t pending_start,
    iree_host_size_t pending_end, bool has_pending) {
  iree_tokenizer_punctuation_match_result_t result;
  memset(&result, 0, sizeof(result));

  switch (behavior) {
    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED:
      // Emit gap, discard match.
      if (gap_end > gap_start) {
        result.segments[0] = gap_start;
        result.segments[1] = gap_end;
        result.segment_count = 1;
      }
      result.last_emit_end = match_end;
      break;

    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED:
      // Emit gap and match as separate segments.
      if (gap_end > gap_start) {
        result.segments[0] = gap_start;
        result.segments[1] = gap_end;
        result.segments[2] = match_start;
        result.segments[3] = match_end;
        result.segment_count = 2;
      } else {
        result.segments[0] = match_start;
        result.segments[1] = match_end;
        result.segment_count = 1;
      }
      result.last_emit_end = match_end;
      break;

    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS:
      // Emit [gap_start, match_end) as one segment.
      result.segments[0] = gap_start;
      result.segments[1] = match_end;
      result.segment_count = 1;
      result.last_emit_end = match_end;
      break;

    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT: {
      // Emit [pending_start or gap_start, gap_end), buffer match for next.
      iree_host_size_t segment_start = has_pending ? pending_start : gap_start;
      if (gap_end > segment_start) {
        result.segments[0] = segment_start;
        result.segments[1] = gap_end;
        result.segment_count = 1;
        result.last_emit_end = gap_end;  // Only advance to what we emitted.
      }
      result.pending_start = match_start;
      result.pending_end = match_end;
      result.has_pending = true;
      // Note: last_emit_end NOT updated if no segment emitted (stays at gap_end
      // or is left unchanged). The pending match bytes will be in
      // remaining_input.
      break;
    }

    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS:
      // Merge consecutive matches, emit gaps between.
      if (has_pending && pending_end == match_start) {
        // Extend pending match (consecutive punctuation).
        result.pending_start = pending_start;
        result.pending_end = match_end;
        result.has_pending = true;
        // last_emit_end not updated - no segment was emitted.
      } else {
        // Emit pending, emit gap, start new pending.
        // Gap starts AFTER pending (if any), not at last_emit_end.
        iree_host_size_t actual_gap_start =
            has_pending ? pending_end : gap_start;
        if (has_pending) {
          result.segments[0] = pending_start;
          result.segments[1] = pending_end;
          result.segment_count = 1;
          result.last_emit_end = pending_end;
        }
        if (gap_end > actual_gap_start) {
          result.segments[result.segment_count * 2] = actual_gap_start;
          result.segments[result.segment_count * 2 + 1] = gap_end;
          result.segment_count++;
          result.last_emit_end = gap_end;
        }
        result.pending_start = match_start;
        result.pending_end = match_end;
        result.has_pending = true;
      }
      break;

    default:
      IREE_ASSERT(false && "invalid punctuation behavior enum value");
      break;
  }
  return result;
}

// Emits all segments from a match result. Returns true if none overflowed.
static inline bool iree_tokenizer_punctuation_emit_result(
    iree_tokenizer_punctuation_emitter_t* emitter,
    const iree_tokenizer_punctuation_match_result_t* result) {
  for (uint8_t i = 0; i < result->segment_count; ++i) {
    iree_tokenizer_punctuation_emit(emitter, result->segments[i * 2],
                                    result->segments[i * 2 + 1]);
  }
  return !emitter->full;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_segmenter_punctuation_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
  iree_tokenizer_regex_split_behavior_t behavior;
} iree_tokenizer_segmenter_punctuation_t;

typedef struct iree_tokenizer_segmenter_punctuation_state_t {
  iree_tokenizer_segmenter_state_t base;

  // Total bytes consumed across all process() calls (absolute position).
  iree_host_size_t bytes_consumed;

  // Position after last emitted segment (absolute byte offset).
  iree_host_size_t last_emit_end;

  // Pending segment for MERGED_WITH_NEXT and CONTIGUOUS behaviors.
  iree_host_size_t pending_start;
  iree_host_size_t pending_end;
  bool has_pending;

  // Cached from segmenter for hot path.
  iree_tokenizer_regex_split_behavior_t behavior;

  // Set after finalize's scan phase completes. Prevents re-running the scan
  // on re-entrant finalize calls (where only trailing flush needs to resume).
  iree_host_size_t finalize_scan_position;
  bool finalize_feed_done;

  // The chunk_base to use for finalize sessions. Set at first finalize call
  // and stays constant until all finalize segments are emitted. This ensures
  // pending segments from process() are emitted with the correct coordinate
  // system. Reset when finalize completes.
  iree_host_size_t finalize_chunk_base;
  // The absolute end position of the finalize input, used by has_pending() to
  // detect when there's trailing data left to emit during finalize.
  iree_host_size_t finalize_input_end;
  bool finalize_chunk_base_set;
} iree_tokenizer_segmenter_punctuation_state_t;

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_punctuation_vtable;

iree_status_t iree_tokenizer_segmenter_punctuation_allocate(
    iree_tokenizer_regex_split_behavior_t behavior, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  iree_tokenizer_segmenter_punctuation_t* segmenter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter));

  iree_tokenizer_segmenter_initialize(
      &segmenter->base, &iree_tokenizer_segmenter_punctuation_vtable,
      sizeof(iree_tokenizer_segmenter_punctuation_state_t));
  segmenter->allocator = allocator;
  segmenter->behavior = behavior;

  *out_segmenter = &segmenter->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_punctuation_destroy(
    iree_tokenizer_segmenter_t* segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_punctuation_t* self =
      (iree_tokenizer_segmenter_punctuation_t*)segmenter;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_punctuation_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_segmenter_punctuation_t* self =
      (const iree_tokenizer_segmenter_punctuation_t*)segmenter;
  iree_tokenizer_segmenter_punctuation_state_t* state =
      (iree_tokenizer_segmenter_punctuation_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.segmenter = segmenter;
  state->behavior = self->behavior;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_punctuation_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Process
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_segmenter_punctuation_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_punctuation_state_t* self =
      (iree_tokenizer_segmenter_punctuation_state_t*)state;

  *out_consumed = 0;
  *out_segment_count = 0;

  if (input.size == 0 || output.capacity == 0) {
    return iree_ok_status();
  }

  iree_host_size_t chunk_base = self->bytes_consumed;

  // Set up emitter with absolute tracking.
  iree_tokenizer_punctuation_emitter_t emitter = {
      .output = output,
      .chunk_base = chunk_base,
      .count = 0,
      .last_end = self->last_emit_end,
      .full = false,
  };

  // Transient copies of mutable state (committed on successful emit).
  iree_host_size_t last_emit_end = self->last_emit_end;
  iree_host_size_t pending_start = self->pending_start;
  iree_host_size_t pending_end = self->pending_end;
  bool has_pending = self->has_pending;

  // Start scanning after pending bytes if they're at the start of this input.
  // The pending segment was already found in a previous call, so we shouldn't
  // re-scan those bytes.
  iree_host_size_t position = 0;
  if (has_pending && pending_end > chunk_base) {
    iree_host_size_t pending_rel_end = pending_end - chunk_base;
    if (pending_rel_end <= input.size) {
      position = pending_rel_end;
    }
  }

  while (position < input.size) {
    iree_host_size_t byte_length = 0;
    bool is_incomplete = false;
    bool is_punct = iree_tokenizer_punctuation_decode_next(
        input.data, position, input.size, &byte_length, &is_incomplete);

    if (is_incomplete) {
      // Incomplete UTF-8 at chunk boundary — stop here.
      break;
    }

    if (is_punct) {
      // Compute absolute positions.
      iree_host_size_t abs_match_start = chunk_base + position;
      iree_host_size_t abs_match_end = abs_match_start + byte_length;

      // Gap is [last_emit_end, abs_match_start).
      iree_tokenizer_punctuation_match_result_t result =
          iree_tokenizer_punctuation_handle_match(
              self->behavior, last_emit_end, abs_match_start, abs_match_start,
              abs_match_end, pending_start, pending_end, has_pending);

      // Emit segments from result. The emitter checks capacity and tracks the
      // end of the last successfully emitted segment for overflow recovery.
      if (!iree_tokenizer_punctuation_emit_result(&emitter, &result)) {
        // Output full. Consume up to the last emitted segment.
        // Discard pending state accumulated after last emit — those bytes are
        // beyond consumption point and will be re-scanned on the next call.
        if (emitter.last_end > chunk_base) {
          *out_consumed = emitter.last_end - chunk_base;
          self->bytes_consumed = emitter.last_end;
          self->last_emit_end = emitter.last_end;
          self->has_pending = false;
        } else {
          // No segments emitted at all (capacity too small for first result).
          *out_consumed = 0;
        }
        *out_segment_count = emitter.count;
        return iree_ok_status();
      }

      // All segments emitted. Commit state from result.
      last_emit_end = result.last_emit_end;
      has_pending = result.has_pending;
      pending_start = result.pending_start;
      pending_end = result.pending_end;
    }

    position += byte_length;
  }

  // Commit callback state to persistent state.
  self->bytes_consumed = chunk_base + position;
  self->last_emit_end = last_emit_end;
  self->has_pending = has_pending;
  self->pending_start = pending_start;
  self->pending_end = pending_end;

  // Determine consumption.
  // The pull-based API contract requires that we only report bytes as consumed
  // if we have emitted complete segments for them. Unconsumed bytes will be
  // passed to finalize() where pending segments can be emitted.
  //
  // When has_pending is true (MERGED_WITH_NEXT, CONTIGUOUS): the pending bytes
  // need to go to finalize() to be emitted, so we consume only up to
  // pending_start (the byte BEFORE the pending segment begins).
  //
  // When has_pending is false but there's trailing data (bytes scanned past
  // the last emitted segment), only report emitted bytes as consumed.
  if (has_pending) {
    // Consume only up to pending_start. The pending bytes [pending_start,
    // pending_end) and any trailing text will be passed to finalize().
    *out_consumed = pending_start - chunk_base;
    self->bytes_consumed = pending_start;
  } else if (self->last_emit_end < self->bytes_consumed) {
    *out_consumed = self->last_emit_end - chunk_base;
    self->bytes_consumed = self->last_emit_end;
  } else {
    *out_consumed = position;
  }

  *out_segment_count = emitter.count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Finalize
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_segmenter_punctuation_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t remaining_input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_punctuation_state_t* self =
      (iree_tokenizer_segmenter_punctuation_state_t*)state;

  // Use a stable chunk_base for the entire finalize session. This is critical
  // when finalize() is called multiple times with the same remaining_input
  // (e.g., when Sequence loops to drain all child segments). Without this,
  // bytes_consumed would accumulate and cause segment offset underflow.
  iree_host_size_t chunk_base;
  if (!self->finalize_chunk_base_set) {
    // First finalize call in this session - capture chunk_base.
    // Use the bytes_consumed value from process(), which corresponds to
    // the coordinate system that pending segments were recorded in.
    // Subtract remaining_input.size to get the base of THIS input.
    chunk_base = self->bytes_consumed > remaining_input.size
                     ? self->bytes_consumed - remaining_input.size
                     : 0;
    // Actually, the correct value depends on whether process() was called.
    // If pending segments exist, they were recorded with their original
    // chunk_base. We need to preserve that relationship.
    // Compute chunk_base to ensure no underflow when emitting segments.
    // chunk_base must be <= pending_start (if has_pending) so that
    // pending_start - chunk_base doesn't underflow.
    // Also, chunk_base + remaining_input.size should be valid.
    //
    // Key insight: pending segments were recorded with ABSOLUTE positions.
    // When emitting, we subtract chunk_base to get relative positions.
    // To avoid underflow, chunk_base must be <= min(pending_start,
    // last_emit_end).
    if (self->has_pending) {
      // chunk_base must be <= pending_start to avoid underflow.
      // Use pending_start as the base - segments will be relative to it.
      chunk_base = self->pending_start;
    } else if (self->last_emit_end < self->bytes_consumed) {
      // There's a trailing gap [last_emit_end, bytes_consumed) to emit.
      chunk_base = self->last_emit_end;
    } else {
      // Nothing to emit from prior state.
      chunk_base = self->bytes_consumed;
    }
    self->finalize_chunk_base = chunk_base;
    self->finalize_input_end = chunk_base + remaining_input.size;
    self->finalize_chunk_base_set = true;
  } else {
    // Subsequent finalize call in same session - reuse saved chunk_base.
    chunk_base = self->finalize_chunk_base;
  }

  iree_host_size_t input_end = self->finalize_input_end;

  iree_tokenizer_punctuation_emitter_t emitter = {
      .output = output,
      .chunk_base = chunk_base,
      .count = 0,
      .last_end = self->last_emit_end,
      .full = false,
  };

  // Phase 1: Scan remaining_input for punctuation matches. This phase runs
  // once; on re-entrant calls (after trailing flush overflow), it is skipped.
  // Start scanning AFTER the pending segment's bytes since those are already
  // tracked. If pending=[chunk_base, pending_end), skip to pending_end.
  if (!self->finalize_feed_done) {
    iree_host_size_t position = self->finalize_scan_position;
    // Skip past pending bytes that are already tracked.
    if (self->has_pending && self->pending_end > chunk_base) {
      iree_host_size_t pending_rel_end = self->pending_end - chunk_base;
      if (position < pending_rel_end) {
        position = pending_rel_end;
      }
    }

    while (position < remaining_input.size) {
      iree_host_size_t byte_length = 0;
      bool is_incomplete = false;
      bool is_punct = iree_tokenizer_punctuation_decode_next(
          remaining_input.data, position, remaining_input.size, &byte_length,
          &is_incomplete);

      if (is_incomplete) {
        // Incomplete UTF-8 at end of input. Include remaining bytes in the
        // trailing gap.
        position = remaining_input.size;
        break;
      }

      if (is_punct) {
        iree_host_size_t abs_match_start = chunk_base + position;
        iree_host_size_t abs_match_end = abs_match_start + byte_length;

        iree_tokenizer_punctuation_match_result_t result =
            iree_tokenizer_punctuation_handle_match(
                self->behavior, self->last_emit_end, abs_match_start,
                abs_match_start, abs_match_end, self->pending_start,
                self->pending_end, self->has_pending);

        if (!iree_tokenizer_punctuation_emit_result(&emitter, &result)) {
          // Output full. Save position for re-entry.
          self->finalize_scan_position = position;
          *out_segment_count = emitter.count;
          return iree_ok_status();
        }

        self->last_emit_end = result.last_emit_end;
        self->has_pending = result.has_pending;
        self->pending_start = result.pending_start;
        self->pending_end = result.pending_end;
      }

      position += byte_length;
    }

    self->finalize_feed_done = true;
  }

  // Phase 2: Flush pending and trailing gap. Each emit may overflow, so state
  // updates are guarded per-emit. On re-entry, self's state reflects what was
  // already emitted (has_pending=false after pending emitted, last_emit_end
  // advanced after each segment).
  iree_host_size_t trailing_end = input_end;

  if (!emitter.full && self->has_pending) {
    if (self->behavior == IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT) {
      // Pending match + trailing gap = one segment.
      iree_tokenizer_punctuation_emit(&emitter, self->pending_start,
                                      trailing_end);
      if (!emitter.full) {
        self->has_pending = false;
        self->last_emit_end = trailing_end;
      }
    } else {
      // CONTIGUOUS: emit pending match, then trailing gap.
      iree_tokenizer_punctuation_emit(&emitter, self->pending_start,
                                      self->pending_end);
      if (!emitter.full) {
        self->has_pending = false;
        self->last_emit_end = self->pending_end;

        // Trailing gap after the pending match.
        iree_tokenizer_punctuation_emit(&emitter, self->last_emit_end,
                                        trailing_end);
        if (!emitter.full) {
          self->last_emit_end = trailing_end;
        }
      }
    }
  } else if (!emitter.full) {
    // Emit trailing gap (non-punctuation text at end of input).
    iree_tokenizer_punctuation_emit(&emitter, self->last_emit_end,
                                    trailing_end);
    if (!emitter.full) {
      self->last_emit_end = trailing_end;
    }
  }

  *out_segment_count = emitter.count;
  if (emitter.full) {
    return iree_ok_status();
  }

  // All phases complete. Reset for potential reuse.
  self->bytes_consumed = input_end;
  self->finalize_scan_position = 0;
  self->finalize_feed_done = false;
  self->finalize_chunk_base_set = false;  // Reset for next finalize session.
  return iree_ok_status();
}

static bool iree_tokenizer_segmenter_punctuation_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  const iree_tokenizer_segmenter_punctuation_state_t* self =
      (const iree_tokenizer_segmenter_punctuation_state_t*)state;
  // Check for pending segment or uncommitted bytes from process().
  if (self->has_pending || self->last_emit_end < self->bytes_consumed) {
    return true;
  }
  // During finalize, also check if there's more to emit before input_end.
  // This handles the case where capacity filled mid-finalize and there's
  // trailing data (e.g., "23" after emitting "." from "1.23").
  if (self->finalize_chunk_base_set &&
      self->last_emit_end < self->finalize_input_end) {
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_punctuation_vtable = {
        .destroy = iree_tokenizer_segmenter_punctuation_destroy,
        .state_initialize =
            iree_tokenizer_segmenter_punctuation_state_initialize,
        .state_deinitialize =
            iree_tokenizer_segmenter_punctuation_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_punctuation_state_process,
        .state_finalize = iree_tokenizer_segmenter_punctuation_state_finalize,
        .state_has_pending =
            iree_tokenizer_segmenter_punctuation_state_has_pending,
};
