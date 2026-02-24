// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/split.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_segmenter_split_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
  iree_tokenizer_regex_split_behavior_t behavior;
  bool invert;
  // false = regex mode, true = literal mode.
  bool is_literal;
  union {
    struct {
      iree_tokenizer_regex_dfa_t dfa;
      iree_tokenizer_regex_stride_t* stride;
      uint8_t* dfa_data;
    } regex;
    struct {
      iree_string_view_t pattern;  // Points into pattern_data
      // Owned copy of pattern bytes.
      uint8_t* pattern_data;
    } literal;
  };
} iree_tokenizer_segmenter_split_t;

typedef struct iree_tokenizer_segmenter_split_state_t {
  iree_tokenizer_segmenter_state_t base;
  iree_tokenizer_regex_exec_state_t regex_state;

  // Position tracking (all absolute byte offsets).
  iree_host_size_t bytes_processed;
  iree_host_size_t last_emit_end;

  // Pending segment for MERGED_WITH_NEXT and CONTIGUOUS behaviors.
  iree_host_size_t pending_start;
  iree_host_size_t pending_end;

  // Cached from segmenter for hot path.
  iree_tokenizer_regex_split_behavior_t behavior;
  bool invert;
  bool is_literal;
  bool has_pending;

  // Set when process() caps consumed due to a pending regex match. The regex
  // state was reset to avoid position underflow, but finalize() will re-scan
  // the unconsumed bytes to emit the deferred match. has_pending() checks this
  // flag to correctly indicate that finalize() is needed.
  bool deferred_to_finalize;

  // Set after finalize's feed+regex phases complete. Prevents re-running them
  // on re-entrant finalize calls (where only trailing flush needs to resume).
  bool finalize_feed_done;

  // Literal mode: tracks partial match position when pattern spans chunks.
  // Value is number of pattern bytes matched so far (0 = no partial match).
  iree_host_size_t literal_match_position;
  // Absolute position where the partial match started.
  iree_host_size_t literal_match_start;
} iree_tokenizer_segmenter_split_state_t;

// Result of processing one match. Handlers are pure functions that compute
// this result; the main loop commits state changes only if all emits succeed.
// Layout optimized: size_t fields first, then small fields packed at end.
typedef struct {
  iree_host_size_t segments[4];  // [start0, end0, start1, end1]
  iree_host_size_t pending_start;
  iree_host_size_t pending_end;
  iree_host_size_t last_emit_end;
  // 0, 1, or 2 segments.
  uint8_t segment_count;
  bool has_pending;
} iree_tokenizer_split_match_result_t;

//===----------------------------------------------------------------------===//
// Behavior Handlers (pure functions, no side effects)
//===----------------------------------------------------------------------===//

// REMOVED: Emit gap (or match if inverted), discard the other.
// Normal mode (invert=false): pattern matches delimiters, emit gaps (content).
// Invert mode (invert=true): pattern matches tokens, emit matches (content).
static iree_tokenizer_split_match_result_t iree_tokenizer_split_handle_removed(
    iree_host_size_t gap_start, iree_host_size_t gap_end,
    iree_host_size_t match_start, iree_host_size_t match_end,
    iree_host_size_t pending_start, iree_host_size_t pending_end,
    bool has_pending, bool invert) {
  (void)pending_start;
  (void)pending_end;
  (void)has_pending;
  iree_tokenizer_split_match_result_t result = {0};
  if (invert) {
    // Invert: emit the match (token), discard the gap (delimiter).
    if (match_end > match_start) {
      result.segments[0] = match_start;
      result.segments[1] = match_end;
      result.segment_count = 1;
    }
  } else {
    // Normal: emit the gap (content), discard the match (delimiter).
    if (gap_end > gap_start) {
      result.segments[0] = gap_start;
      result.segments[1] = gap_end;
      result.segment_count = 1;
    }
  }
  result.last_emit_end = match_end;
  return result;
}

// ISOLATED: Emit gap and match as separate segments.
// Order is always gap first, then match (preserves positional order).
// The invert flag doesn't change the behavior - both are always emitted.
static iree_tokenizer_split_match_result_t iree_tokenizer_split_handle_isolated(
    iree_host_size_t gap_start, iree_host_size_t gap_end,
    iree_host_size_t match_start, iree_host_size_t match_end,
    iree_host_size_t pending_start, iree_host_size_t pending_end,
    bool has_pending, bool invert) {
  (void)pending_start;
  (void)pending_end;
  (void)has_pending;
  (void)invert;  // ISOLATED emits both regardless of invert.

  iree_tokenizer_split_match_result_t result = {0};
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
  return result;
}

// MERGED_WITH_PREVIOUS: Emit [gap_start, match_end) as one segment.
// Merges the content before a delimiter with the delimiter itself.
// In invert mode, this merges the delimiter with the following token.
static iree_tokenizer_split_match_result_t
iree_tokenizer_split_handle_merged_with_previous(iree_host_size_t gap_start,
                                                 iree_host_size_t match_end,
                                                 iree_host_size_t pending_start,
                                                 iree_host_size_t pending_end,
                                                 bool has_pending,
                                                 bool invert) {
  (void)pending_start;
  (void)pending_end;
  (void)has_pending;
  (void)invert;  // Same behavior: always merge [gap_start, match_end).
  iree_tokenizer_split_match_result_t result = {0};
  result.segments[0] = gap_start;
  result.segments[1] = match_end;
  result.segment_count = 1;
  result.last_emit_end = match_end;
  return result;
}

// MERGED_WITH_NEXT: Prepend pending to gap, buffer match for next.
// Normal: buffer delimiter (match), prepend to next content (gap).
// Invert: buffer delimiter (gap), prepend to next content (match).
static iree_tokenizer_split_match_result_t
iree_tokenizer_split_handle_merged_with_next(iree_host_size_t gap_start,
                                             iree_host_size_t gap_end,
                                             iree_host_size_t match_start,
                                             iree_host_size_t match_end,
                                             iree_host_size_t pending_start,
                                             bool has_pending, bool invert) {
  iree_tokenizer_split_match_result_t result = {0};
  if (invert) {
    // Invert: emit pending+match (content), buffer gap (delimiter).
    iree_host_size_t seg_start = has_pending ? pending_start : match_start;
    if (match_end > seg_start) {
      result.segments[0] = seg_start;
      result.segments[1] = match_end;
      result.segment_count = 1;
    }
    result.pending_start = gap_start;
    result.pending_end = gap_end;
    result.has_pending = (gap_end > gap_start);
  } else {
    // Normal: emit pending+gap (content), buffer match (delimiter).
    iree_host_size_t seg_start = has_pending ? pending_start : gap_start;
    if (gap_end > seg_start) {
      result.segments[0] = seg_start;
      result.segments[1] = gap_end;
      result.segment_count = 1;
    }
    result.pending_start = match_start;
    result.pending_end = match_end;
    result.has_pending = true;
  }
  result.last_emit_end = match_end;
  return result;
}

// CONTIGUOUS: Merge consecutive delimiters, emit content between.
// Normal: merge consecutive matches (delimiters), emit gaps (content).
// Invert: merge consecutive gaps (delimiters), emit matches (content).
static iree_tokenizer_split_match_result_t
iree_tokenizer_split_handle_contiguous(iree_host_size_t gap_start,
                                       iree_host_size_t gap_end,
                                       iree_host_size_t match_start,
                                       iree_host_size_t match_end,
                                       iree_host_size_t pending_start,
                                       iree_host_size_t pending_end,
                                       bool has_pending, bool invert) {
  iree_tokenizer_split_match_result_t result = {0};
  if (invert) {
    // Invert: merge consecutive gaps (delimiters), emit matches (content).
    if (has_pending && pending_end == gap_start && gap_end > gap_start) {
      // Extend pending gap (consecutive delimiters).
      result.pending_start = pending_start;
      result.pending_end = gap_end;
      result.has_pending = true;
    } else {
      // Emit pending (merged delimiters), emit match (content), buffer gap.
      if (has_pending) {
        result.segments[0] = pending_start;
        result.segments[1] = pending_end;
        result.segment_count = 1;
      }
      if (match_end > match_start) {
        result.segments[result.segment_count * 2] = match_start;
        result.segments[result.segment_count * 2 + 1] = match_end;
        result.segment_count++;
      }
      if (gap_end > gap_start) {
        result.pending_start = gap_start;
        result.pending_end = gap_end;
        result.has_pending = true;
      }
    }
  } else {
    // Normal: merge consecutive matches (delimiters), emit gaps (content).
    if (has_pending && pending_end == match_start) {
      // Extend pending match (consecutive delimiters).
      result.pending_start = pending_start;
      result.pending_end = match_end;
      result.has_pending = true;
    } else {
      // Emit pending (merged delimiters), emit gap (content), buffer match.
      if (has_pending) {
        result.segments[0] = pending_start;
        result.segments[1] = pending_end;
        result.segment_count = 1;
      }
      if (gap_end > gap_start) {
        result.segments[result.segment_count * 2] = gap_start;
        result.segments[result.segment_count * 2 + 1] = gap_end;
        result.segment_count++;
      }
      result.pending_start = match_start;
      result.pending_end = match_end;
      result.has_pending = true;
    }
  }
  result.last_emit_end = match_end;
  return result;
}

//===----------------------------------------------------------------------===//
// Split Processing
//===----------------------------------------------------------------------===//

// Dispatches to the appropriate behavior handler.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline iree_tokenizer_split_match_result_t
iree_tokenizer_split_handle_match(
    iree_tokenizer_regex_split_behavior_t behavior, iree_host_size_t gap_start,
    iree_host_size_t gap_end, iree_host_size_t match_start,
    iree_host_size_t match_end, iree_host_size_t pending_start,
    iree_host_size_t pending_end, bool has_pending, bool invert) {
  switch (behavior) {
    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED:
      return iree_tokenizer_split_handle_removed(
          gap_start, gap_end, match_start, match_end, pending_start,
          pending_end, has_pending, invert);
    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED:
      return iree_tokenizer_split_handle_isolated(
          gap_start, gap_end, match_start, match_end, pending_start,
          pending_end, has_pending, invert);
    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS:
      return iree_tokenizer_split_handle_merged_with_previous(
          gap_start, match_end, pending_start, pending_end, has_pending,
          invert);
    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT:
      return iree_tokenizer_split_handle_merged_with_next(
          gap_start, gap_end, match_start, match_end, pending_start,
          has_pending, invert);
    case IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS:
      return iree_tokenizer_split_handle_contiguous(
          gap_start, gap_end, match_start, match_end, pending_start,
          pending_end, has_pending, invert);
    default: {
      IREE_ASSERT(false && "invalid split behavior enum value");
      iree_tokenizer_split_match_result_t result = {0};
      return result;
    }
  }
}

typedef struct {
  iree_tokenizer_segment_output_t output;
  iree_host_size_t chunk_base;
  iree_host_size_t count;
  // End of last successfully emitted segment.
  iree_host_size_t last_end;
  bool full;
} iree_tokenizer_split_emitter_t;

// Emits a segment. Empty segments (start >= end) are skipped.
// Sets emitter->full on overflow; caller should check after batch of emits.
static inline void iree_tokenizer_split_emit(
    iree_tokenizer_split_emitter_t* emitter, iree_host_size_t start,
    iree_host_size_t end, bool absolute) {
  if (start >= end || emitter->full) return;
  if (emitter->count >= emitter->output.capacity) {
    emitter->full = true;
    return;
  }
  iree_host_size_t offset = absolute ? 0 : emitter->chunk_base;
  emitter->output.values[emitter->count].start = start - offset;
  emitter->output.values[emitter->count].end = end - offset;
  emitter->count++;
  emitter->last_end = end;
}

// Emits all segments from a match result. Returns true if none overflowed.
static inline bool iree_tokenizer_split_emit_result(
    iree_tokenizer_split_emitter_t* emitter,
    const iree_tokenizer_split_match_result_t* result, bool absolute) {
  for (uint8_t i = 0; i < result->segment_count; ++i) {
    iree_tokenizer_split_emit(emitter, result->segments[i * 2],
                              result->segments[i * 2 + 1], absolute);
  }
  return !emitter->full;
}

// Context for the inline callback, shared between process() and finalize().
// Holds a transient copy of mutable state that is committed back to the
// persistent state struct after the regex feed/finalize completes.
typedef struct iree_tokenizer_split_callback_context_t {
  iree_tokenizer_split_emitter_t* emitter;
  // Mutable state (updated per-match on successful emit).
  iree_host_size_t last_emit_end;
  iree_host_size_t pending_start;
  iree_host_size_t pending_end;
  bool has_pending;
  // Immutable config.
  iree_tokenizer_regex_split_behavior_t behavior;
  bool invert;
  bool absolute;
} iree_tokenizer_split_callback_context_t;

// Processes each regex match inline: computes behavior result, emits segments,
// and commits state. Returns RESOURCE_EXHAUSTED if output capacity is
// insufficient (the match is not committed and the regex will stop).
static inline iree_status_t iree_tokenizer_split_inline_callback(
    void* user_data, iree_tokenizer_regex_match_t match) {
  iree_tokenizer_split_callback_context_t* context =
      (iree_tokenizer_split_callback_context_t*)user_data;

  // Compute behavior result (pure, no side effects).
  // Gap is the region between the last emit and the current match start.
  // Match is the matched region itself.
  // The invert flag is passed to handlers so they can swap semantics internally
  // (e.g., REMOVED emits match instead of gap when inverted).
  iree_tokenizer_split_match_result_t result =
      iree_tokenizer_split_handle_match(
          context->behavior, context->last_emit_end, match.start, match.start,
          match.end, context->pending_start, context->pending_end,
          context->has_pending, context->invert);

  // Emit segments. If the emitter fills mid-result, the partially-emitted
  // segments are committed (emitter.last_end tracks progress), and this match's
  // full state changes are NOT committed to context (ensuring re-scan picks
  // up the un-emitted segments).
  if (!iree_tokenizer_split_emit_result(context->emitter, &result,
                                        context->absolute)) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  // All segments emitted successfully. Commit state.
  context->last_emit_end = result.last_emit_end;
  context->has_pending = result.has_pending;
  context->pending_start = result.pending_start;
  context->pending_end = result.pending_end;

  return iree_ok_status();
}

// Processes input with literal string matching. Returns a status and updates
// emitter/context state. Uses the same callback context structure as regex
// mode for consistency.
//
// The literal matching algorithm:
// 1. If we have a partial match from previous chunk, try to continue it.
// 2. Scan for new matches byte-by-byte.
// 3. Track partial matches at end of chunk that may continue next time.
static iree_status_t iree_tokenizer_split_literal_process(
    const iree_tokenizer_segmenter_split_t* segmenter,
    iree_tokenizer_segmenter_split_state_t* state, iree_string_view_t input,
    iree_host_size_t chunk_base,
    iree_tokenizer_split_callback_context_t* context) {
  const char* pattern = segmenter->literal.pattern.data;
  iree_host_size_t pattern_length = segmenter->literal.pattern.size;

  iree_host_size_t scan_position = 0;

  // Resume partial match from previous chunk if any.
  if (state->literal_match_position > 0) {
    iree_host_size_t remaining = pattern_length - state->literal_match_position;
    iree_host_size_t can_check =
        (remaining < input.size) ? remaining : input.size;

    // Check if the partial match continues.
    if (memcmp(input.data, pattern + state->literal_match_position,
               can_check) == 0) {
      if (can_check == remaining) {
        // Full match completed! Report the match.
        iree_host_size_t match_start = state->literal_match_start;
        iree_host_size_t match_end = chunk_base + can_check;
        iree_tokenizer_regex_match_t match = {match_start, match_end};
        iree_status_t status =
            iree_tokenizer_split_inline_callback(context, match);
        if (!iree_status_is_ok(status)) {
          // Reset partial match state since we're restarting.
          state->literal_match_position = 0;
          return status;
        }
        scan_position = can_check;
        state->literal_match_position = 0;
      } else {
        // Still partial - need more input.
        state->literal_match_position += can_check;
        return iree_ok_status();
      }
    } else {
      // Partial match failed. We need to re-scan from after the false start.
      // The bytes from literal_match_start are already consumed by previous
      // chunks, so we just reset and continue scanning this chunk.
      state->literal_match_position = 0;
    }
  }

  // Scan for new matches.
  while (scan_position < input.size) {
    // Look for the first character of the pattern.
    const char* found = memchr(input.data + scan_position, pattern[0],
                               input.size - scan_position);
    if (!found) {
      // No more potential matches in this chunk.
      break;
    }

    iree_host_size_t match_offset = (iree_host_size_t)(found - input.data);
    iree_host_size_t remaining_input = input.size - match_offset;

    if (remaining_input >= pattern_length) {
      // Enough bytes to check full pattern.
      if (memcmp(found, pattern, pattern_length) == 0) {
        // Full match found.
        iree_host_size_t match_start = chunk_base + match_offset;
        iree_host_size_t match_end = match_start + pattern_length;
        iree_tokenizer_regex_match_t match = {match_start, match_end};
        iree_status_t status =
            iree_tokenizer_split_inline_callback(context, match);
        if (!iree_status_is_ok(status)) {
          return status;
        }
        scan_position = match_offset + pattern_length;
      } else {
        // Not a match, continue from next byte.
        scan_position = match_offset + 1;
      }
    } else {
      // Potential partial match at end of chunk.
      if (memcmp(found, pattern, remaining_input) == 0) {
        // Partial match - save state for next chunk.
        state->literal_match_start = chunk_base + match_offset;
        state->literal_match_position = remaining_input;
      }
      // Either way, we're done with this chunk.
      break;
    }
  }

  return iree_ok_status();
}

// Finalizes literal matching. Handles any partial match from the last chunk.
static iree_status_t iree_tokenizer_split_literal_finalize(
    const iree_tokenizer_segmenter_split_t* segmenter,
    iree_tokenizer_segmenter_split_state_t* state,
    iree_string_view_t remaining_input, iree_host_size_t chunk_base,
    iree_tokenizer_split_callback_context_t* context) {
  const char* pattern = segmenter->literal.pattern.data;
  iree_host_size_t pattern_length = segmenter->literal.pattern.size;

  // Resume partial match if any.
  if (state->literal_match_position > 0) {
    iree_host_size_t remaining = pattern_length - state->literal_match_position;
    if (remaining_input.size >= remaining &&
        memcmp(remaining_input.data, pattern + state->literal_match_position,
               remaining) == 0) {
      // Full match completed.
      iree_host_size_t match_start = state->literal_match_start;
      iree_host_size_t match_end = chunk_base + remaining;
      iree_tokenizer_regex_match_t match = {match_start, match_end};
      iree_status_t status =
          iree_tokenizer_split_inline_callback(context, match);
      if (!iree_status_is_ok(status)) {
        state->literal_match_position = 0;
        return status;
      }
      // Continue scanning after the match.
      remaining_input.data += remaining;
      remaining_input.size -= remaining;
      chunk_base += remaining;
    }
    state->literal_match_position = 0;
  }

  // Scan remaining input for matches.
  iree_host_size_t scan_position = 0;
  while (scan_position < remaining_input.size) {
    const char* found = memchr(remaining_input.data + scan_position, pattern[0],
                               remaining_input.size - scan_position);
    if (!found) break;

    iree_host_size_t match_offset =
        (iree_host_size_t)(found - remaining_input.data);
    iree_host_size_t bytes_left = remaining_input.size - match_offset;

    if (bytes_left >= pattern_length &&
        memcmp(found, pattern, pattern_length) == 0) {
      iree_host_size_t match_start = chunk_base + match_offset;
      iree_host_size_t match_end = match_start + pattern_length;
      iree_tokenizer_regex_match_t match = {match_start, match_end};
      iree_status_t status =
          iree_tokenizer_split_inline_callback(context, match);
      if (!iree_status_is_ok(status)) {
        return status;
      }
      scan_position = match_offset + pattern_length;
    } else {
      scan_position = match_offset + 1;
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable Implementation
//===----------------------------------------------------------------------===//

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_split_vtable;

iree_status_t iree_tokenizer_segmenter_split_allocate(
    iree_tokenizer_regex_dfa_t dfa, uint8_t* dfa_data,
    iree_tokenizer_regex_split_behavior_t behavior, bool invert,
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(dfa_data);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  iree_tokenizer_segmenter_split_t* segmenter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter));

  iree_tokenizer_segmenter_initialize(
      &segmenter->base, &iree_tokenizer_segmenter_split_vtable,
      sizeof(iree_tokenizer_segmenter_split_state_t));
  segmenter->allocator = allocator;
  segmenter->behavior = behavior;
  segmenter->invert = invert;
  segmenter->is_literal = false;
  segmenter->regex.dfa = dfa;
  segmenter->regex.dfa_data = dfa_data;

  // Compute stride acceleration data for the DFA.
  iree_status_t status = iree_tokenizer_regex_stride_allocate(
      &segmenter->regex.dfa, allocator, &segmenter->regex.stride);

  if (iree_status_is_ok(status)) {
    *out_segmenter = &segmenter->base;
  } else {
    iree_allocator_free(allocator, dfa_data);
    iree_allocator_free(allocator, segmenter);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_segmenter_split_literal_allocate(
    iree_string_view_t pattern, iree_tokenizer_regex_split_behavior_t behavior,
    bool invert, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  if (pattern.size == 0) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "literal split pattern cannot be empty"));
  }

  iree_tokenizer_segmenter_split_t* segmenter = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter);

  // Allocate and copy pattern data.
  uint8_t* pattern_data = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(allocator, pattern.size, (void**)&pattern_data);
  }

  if (iree_status_is_ok(status)) {
    memcpy(pattern_data, pattern.data, pattern.size);
    iree_tokenizer_segmenter_initialize(
        &segmenter->base, &iree_tokenizer_segmenter_split_vtable,
        sizeof(iree_tokenizer_segmenter_split_state_t));
    segmenter->allocator = allocator;
    segmenter->behavior = behavior;
    segmenter->invert = invert;
    segmenter->is_literal = true;
    segmenter->literal.pattern_data = pattern_data;
    segmenter->literal.pattern =
        iree_make_string_view((const char*)pattern_data, pattern.size);
    *out_segmenter = &segmenter->base;
  } else {
    iree_allocator_free(allocator, pattern_data);
    iree_allocator_free(allocator, segmenter);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_tokenizer_segmenter_split_destroy(
    iree_tokenizer_segmenter_t* segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_split_t* self =
      (iree_tokenizer_segmenter_split_t*)segmenter;
  iree_allocator_t allocator = self->allocator;
  if (self->is_literal) {
    if (self->literal.pattern_data) {
      iree_allocator_free(allocator, self->literal.pattern_data);
    }
  } else {
    iree_tokenizer_regex_stride_free(self->regex.stride, allocator);
    if (self->regex.dfa_data) {
      iree_allocator_free(allocator, self->regex.dfa_data);
    }
  }
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_split_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_segmenter_split_t* self =
      (const iree_tokenizer_segmenter_split_t*)segmenter;
  iree_tokenizer_segmenter_split_state_t* state =
      (iree_tokenizer_segmenter_split_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.segmenter = segmenter;
  state->behavior = self->behavior;
  state->invert = self->invert;
  state->is_literal = self->is_literal;
  if (!self->is_literal) {
    iree_tokenizer_regex_exec_initialize(&state->regex_state, &self->regex.dfa);
  }

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_split_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_split_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_split_state_t* self =
      (iree_tokenizer_segmenter_split_state_t*)state;
  const iree_tokenizer_segmenter_split_t* segmenter =
      (const iree_tokenizer_segmenter_split_t*)state->segmenter;

  *out_consumed = 0;
  *out_segment_count = 0;

  if (input.size == 0 || output.capacity == 0) {
    return iree_ok_status();
  }

  iree_host_size_t chunk_base = self->bytes_processed;

  // Set up emitter and callback context. The context holds a transient copy
  // of mutable state that the inline callback updates per-match.
  iree_tokenizer_split_emitter_t emitter = {
      .output = output,
      .chunk_base = chunk_base,
      .count = 0,
      .last_end = self->last_emit_end,
      .full = false,
  };
  iree_tokenizer_split_callback_context_t context = {
      .emitter = &emitter,
      .last_emit_end = self->last_emit_end,
      .pending_start = self->pending_start,
      .pending_end = self->pending_end,
      .has_pending = self->has_pending,
      .behavior = self->behavior,
      .invert = self->invert,
      .absolute = false,
  };

  // Feed pattern matcher; matches are processed inline via callback.
  iree_status_t feed_status;
  if (self->is_literal) {
    feed_status = iree_tokenizer_split_literal_process(segmenter, self, input,
                                                       chunk_base, &context);
  } else {
    feed_status = iree_tokenizer_regex_exec_feed(
        &segmenter->regex.dfa, &self->regex_state, input, chunk_base,
        segmenter->regex.stride, iree_tokenizer_split_inline_callback,
        &context);
  }
  if (!iree_status_is_ok(feed_status) &&
      !iree_status_is_resource_exhausted(feed_status)) {
    return feed_status;
  }
  if (iree_status_is_resource_exhausted(feed_status)) {
    emitter.full = true;
  }
  iree_status_ignore(feed_status);

  // Determine consumption and commit state based on output capacity.
  if (emitter.full && emitter.last_end > chunk_base) {
    // Output filled mid-chunk: consume up to last emitted segment.
    // Discard any pending state accumulated after the last emit — those bytes
    // are beyond our consumption point and will be re-fed on the next call.
    *out_consumed = emitter.last_end - chunk_base;
    self->bytes_processed = emitter.last_end;
    self->last_emit_end = emitter.last_end;
    self->has_pending = false;
    self->literal_match_position = 0;
    if (!self->is_literal) {
      iree_tokenizer_regex_exec_initialize(&self->regex_state,
                                           &segmenter->regex.dfa);
    }
  } else if (emitter.full) {
    // Output filled but no progress (capacity too small for first result).
    // Leave persistent state unchanged — nothing was consumed or emitted.
    *out_consumed = 0;
    self->literal_match_position = 0;
    if (!self->is_literal) {
      iree_tokenizer_regex_exec_initialize(&self->regex_state,
                                           &segmenter->regex.dfa);
    }
  } else {
    // Normal case: consume all input, commit full callback state.
    self->bytes_processed = chunk_base + input.size;
    self->last_emit_end = context.last_emit_end;
    self->has_pending = context.has_pending;
    self->pending_start = context.pending_start;
    self->pending_end = context.pending_end;

    // Check if the regex has a pending match that finalize() will need to emit.
    // A pending regex match exists when in_match is true AND there's an
    // accepting position (either lookahead-passed or fallback).
    //
    // CRITICAL: When there's a pending regex match, we must NOT consume past
    // last_emit_end. The regex tracks match positions cumulatively from init,
    // but finalize() uses chunk_base=bytes_processed for position arithmetic.
    // If we consume past the pending match's start position, finalize() will
    // compute start - chunk_base = underflow.
    //
    // Example: Input "    if" matches [0,3) and [3,6). The [3,6) match is
    // pending after process() (greedy match waits for confirming byte).
    // If we set bytes_processed=6, finalize() gets chunk_base=6 and computes
    // 3 - 6 = underflow. By capping at last_emit_end=3, finalize() gets
    // chunk_base=3 and re-scans " if", producing the correct [0,3) relative
    // position (absolute [3,6)).
    bool has_pending_regex_match =
        !self->is_literal && self->regex_state.in_match &&
        (self->regex_state.has_accept || self->regex_state.has_accept_fallback);

    // Check if we should use wraparound arithmetic for this behavior.
    // ISOLATED+invert needs wraparound because trailing gaps must be emitted
    // and has_pending relies on last_emit_end < bytes_processed.
    bool uses_wraparound =
        self->invert &&
        self->behavior == IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED;

    // Determine if we need to cap consumed bytes:
    // 1. Pending regex match: ALWAYS cap to avoid position underflow
    // 2. Trailing data: cap when not handled by pending segment logic
    bool has_trailing_data = self->last_emit_end < self->bytes_processed;
    bool should_cap_trailing =
        !self->has_pending && !uses_wraparound && has_trailing_data;

    if (has_pending_regex_match || should_cap_trailing) {
      // Cap consumed at last emitted segment. The unconsumed bytes will be
      // passed to finalize(), which can re-scan them with correct chunk_base.
      *out_consumed = self->last_emit_end - chunk_base;
      self->bytes_processed = self->last_emit_end;
      self->literal_match_position = 0;
      if (!self->is_literal) {
        iree_tokenizer_regex_exec_initialize(&self->regex_state,
                                             &segmenter->regex.dfa);
      }
      // Set deferred flag only when capping specifically for the pending regex
      // match case that previously used wraparound arithmetic. This preserves
      // has_pending()=true for ISOLATED+invert mode, which relies on it.
      // For other cases (should_cap_trailing), the original behavior was
      // has_pending()=false after capping.
      self->deferred_to_finalize = has_pending_regex_match && uses_wraparound;
    } else {
      *out_consumed = input.size;
      self->deferred_to_finalize = false;
    }
  }

  *out_segment_count = emitter.count;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_segmenter_split_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t remaining_input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_split_state_t* self =
      (iree_tokenizer_segmenter_split_state_t*)state;
  const iree_tokenizer_segmenter_split_t* segmenter =
      (const iree_tokenizer_segmenter_split_t*)state->segmenter;

  *out_segment_count = 0;

  iree_host_size_t chunk_base = self->bytes_processed;

  // Set up emitter and callback context (same pattern as process).
  iree_tokenizer_split_emitter_t emitter = {
      .output = output,
      .chunk_base = chunk_base,
      .count = 0,
      .last_end = self->last_emit_end,
      .full = false,
  };
  iree_tokenizer_split_callback_context_t context = {
      .emitter = &emitter,
      .last_emit_end = self->last_emit_end,
      .pending_start = self->pending_start,
      .pending_end = self->pending_end,
      .has_pending = self->has_pending,
      .behavior = self->behavior,
      .invert = self->invert,
      .absolute = false,
  };

  iree_host_size_t input_end = chunk_base + remaining_input.size;

  // Phase 1: Feed remaining input and finalize pattern matcher. This phase
  // runs once; on re-entrant calls (after trailing flush overflow), it is
  // skipped.
  if (!self->finalize_feed_done) {
    if (self->is_literal) {
      // Literal mode: finalize handles partial matches and remaining input.
      iree_status_t finalize_status = iree_tokenizer_split_literal_finalize(
          segmenter, self, remaining_input, chunk_base, &context);
      if (iree_status_is_resource_exhausted(finalize_status)) {
        // Output full - translate to emitter.full flag.
        iree_status_ignore(finalize_status);
        emitter.full = true;
      } else if (!iree_status_is_ok(finalize_status)) {
        return finalize_status;
      }
    } else {
      // Regex mode: feed remaining input through regex.
      if (remaining_input.size > 0) {
        iree_status_t feed_status = iree_tokenizer_regex_exec_feed(
            &segmenter->regex.dfa, &self->regex_state, remaining_input,
            chunk_base, segmenter->regex.stride,
            iree_tokenizer_split_inline_callback, &context);
        if (iree_status_is_resource_exhausted(feed_status)) {
          // Output full during feed - preserve regex state for re-entry.
          iree_status_ignore(feed_status);
          emitter.full = true;
        } else if (!iree_status_is_ok(feed_status)) {
          return feed_status;
        }
      }

      // Finalize regex (may produce one final match at end-of-input).
      // Skip if output already full.
      if (!emitter.full) {
        iree_status_t finalize_status = iree_tokenizer_regex_exec_finalize(
            &segmenter->regex.dfa, &self->regex_state, input_end,
            iree_tokenizer_split_inline_callback, &context);
        if (iree_status_is_ok(finalize_status)) {
          iree_tokenizer_regex_exec_initialize(&self->regex_state,
                                               &segmenter->regex.dfa);
        } else if (iree_status_is_resource_exhausted(finalize_status)) {
          // Final match couldn't be emitted - preserve regex state for
          // re-entry.
          iree_status_ignore(finalize_status);
          emitter.full = true;
        } else {
          return finalize_status;
        }
      }
    }

    // Commit progress. When full, commit partial progress so re-entry doesn't
    // re-emit. When complete, commit full progress and mark phase done.
    if (emitter.full) {
      self->last_emit_end = emitter.last_end;
    } else {
      self->last_emit_end = context.last_emit_end;
      self->has_pending = context.has_pending;
      self->pending_start = context.pending_start;
      self->pending_end = context.pending_end;
      self->finalize_feed_done = true;
      self->deferred_to_finalize = false;  // Deferred regex work is done.
    }
  }

  // Phase 2: Flush pending and trailing gap. Each emit may overflow, so state
  // updates are guarded per-emit. On re-entry, self's state reflects what was
  // already emitted (has_pending=false after pending emitted, last_emit_end
  // advanced after each segment).
  //
  // Trailing gap emission depends on mode:
  // - Normal mode: always emit trailing gap (it's content)
  // - Invert mode + ISOLATED: emit trailing gap (ISOLATED emits everything)
  // - Invert mode + others: skip trailing gap (it's a delimiter)
  bool emit_trailing_gap =
      !self->invert ||
      self->behavior == IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED;
  iree_host_size_t trailing_end = input_end;
  if (!emitter.full && self->has_pending) {
    if (self->behavior == IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT) {
      // MERGED_WITH_NEXT: merge pending delimiter with trailing content.
      // In invert mode, pending is a delimiter (gap), trailing is content.
      // Only merge if there's trailing content OR emit_trailing_gap is true.
      if (emit_trailing_gap || trailing_end > self->pending_end) {
        iree_tokenizer_split_emit(&emitter, self->pending_start, trailing_end,
                                  false);
      } else {
        // Invert mode: pending is delimiter with no content to merge, discard.
        iree_tokenizer_split_emit(&emitter, self->pending_start,
                                  self->pending_end, false);
      }
      if (!emitter.full) {
        self->has_pending = false;
        self->last_emit_end = trailing_end;
      }
    } else {
      iree_tokenizer_split_emit(&emitter, self->pending_start,
                                self->pending_end, false);
      if (!emitter.full) {
        self->has_pending = false;
        self->last_emit_end = self->pending_end;
        if (emit_trailing_gap) {
          iree_tokenizer_split_emit(&emitter, self->last_emit_end, trailing_end,
                                    false);
        }
        if (!emitter.full) {
          self->last_emit_end = trailing_end;
        }
      }
    }
  } else if (!emitter.full) {
    if (emit_trailing_gap) {
      iree_tokenizer_split_emit(&emitter, self->last_emit_end, trailing_end,
                                false);
    }
    if (!emitter.full) {
      self->last_emit_end = trailing_end;
    }
  }

  *out_segment_count = emitter.count;
  if (emitter.full) {
    // Commit partial progress so re-entry doesn't re-emit. Caller should check
    // has_pending() to know if re-entry is needed.
    self->last_emit_end = emitter.last_end;
    return iree_ok_status();
  }
  // All phases complete. Advance bytes_processed and reset for potential reuse.
  self->bytes_processed = input_end;
  self->finalize_feed_done = false;
  return iree_ok_status();
}

static bool iree_tokenizer_segmenter_split_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  const iree_tokenizer_segmenter_split_state_t* self =
      (const iree_tokenizer_segmenter_split_state_t*)state;

  // Check if there's a buffered pending segment.
  if (self->has_pending) return true;

  // Check if process() capped consumed due to a pending regex match.
  // The regex state was reset, but finalize() will re-scan the unconsumed
  // bytes to emit the deferred match.
  if (self->deferred_to_finalize) return true;

  // Check if the regex has a partial match that finalize() will complete.
  // This is critical for patterns like \w+ where the match extends to end of
  // input but the regex stays in "in_match" state waiting for more input.
  if (!self->is_literal && self->regex_state.in_match &&
      (self->regex_state.has_accept || self->regex_state.has_accept_fallback)) {
    return true;
  }

  // Check if finalize() will emit trailing data.
  // In normal mode, trailing gaps (content) are emitted.
  // In invert mode, trailing gaps (delimiters) are only emitted for ISOLATED.
  if (self->last_emit_end < self->bytes_processed) {
    bool emit_trailing =
        !self->invert ||
        self->behavior == IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED;
    return emit_trailing;
  }

  return false;
}

static iree_status_t iree_tokenizer_segmenter_split_state_flush(
    iree_tokenizer_segmenter_state_t* state,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_segment_count,
    iree_host_size_t* out_bytes_committed) {
  iree_tokenizer_segmenter_split_state_t* self =
      (iree_tokenizer_segmenter_split_state_t*)state;
  const iree_tokenizer_segmenter_split_t* segmenter =
      (const iree_tokenizer_segmenter_split_t*)state->segmenter;
  *out_segment_count = 0;
  *out_bytes_committed = self->last_emit_end;

  if (output.capacity == 0) {
    return iree_ok_status();
  }

  iree_tokenizer_split_emitter_t emitter = {
      .output = output,
      .chunk_base = 0,  // Segments are absolute buffer positions.
      .count = 0,
      .last_end = self->last_emit_end,
      .full = false,
  };

  // If we have an accepted match, emit it first.
  if (self->regex_state.in_match &&
      (self->regex_state.has_accept || self->regex_state.has_accept_fallback)) {
    iree_host_size_t match_end = self->regex_state.has_accept
                                     ? self->regex_state.last_accept
                                     : self->regex_state.last_accept_fallback;

    // Gap is between last emit and match start, match is the regex match.
    // Invert flag is passed to handler to swap semantics internally.
    iree_tokenizer_split_match_result_t result =
        iree_tokenizer_split_handle_match(
            self->behavior, self->last_emit_end, self->regex_state.match_start,
            self->regex_state.match_start, match_end, self->pending_start,
            self->pending_end, self->has_pending, self->invert);

    if (!iree_tokenizer_split_emit_result(&emitter, &result, true)) {
      *out_segment_count = emitter.count;
      *out_bytes_committed = emitter.last_end;
      return iree_ok_status();
    }

    self->last_emit_end = result.last_emit_end;
    self->has_pending = result.has_pending;
    self->pending_start = result.pending_start;
    self->pending_end = result.pending_end;
  }

  // Emit trailing gap up to bytes_processed, like finalize does.
  // This ensures we commit all scanned bytes so the buffer can compact.
  // State updates are guarded per-emit for correctness of out_bytes_committed.
  //
  // Trailing gap emission depends on mode (same logic as finalize).
  bool emit_trailing_gap =
      !self->invert ||
      self->behavior == IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED;
  iree_host_size_t trailing_end = self->bytes_processed;
  if (!emitter.full && trailing_end > self->last_emit_end) {
    if (self->has_pending) {
      if (self->behavior == IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT) {
        if (emit_trailing_gap || trailing_end > self->pending_end) {
          iree_tokenizer_split_emit(&emitter, self->pending_start, trailing_end,
                                    false);
        } else {
          iree_tokenizer_split_emit(&emitter, self->pending_start,
                                    self->pending_end, false);
        }
        if (!emitter.full) {
          self->has_pending = false;
          self->last_emit_end = trailing_end;
        }
      } else {
        iree_tokenizer_split_emit(&emitter, self->pending_start,
                                  self->pending_end, false);
        if (!emitter.full) {
          self->has_pending = false;
          self->last_emit_end = self->pending_end;
          if (emit_trailing_gap) {
            iree_tokenizer_split_emit(&emitter, self->last_emit_end,
                                      trailing_end, false);
          }
          if (!emitter.full) {
            self->last_emit_end = trailing_end;
          }
        }
      }
    } else {
      if (emit_trailing_gap) {
        iree_tokenizer_split_emit(&emitter, self->last_emit_end, trailing_end,
                                  false);
      }
      if (!emitter.full) {
        self->last_emit_end = trailing_end;
      }
    }
  }

  // Check if output is full BEFORE resetting state. This matches finalize()'s
  // pattern: only reset state when all emissions succeeded. If full, return
  // partial results and preserve state so caller can retry with more capacity.
  *out_segment_count = emitter.count;
  *out_bytes_committed = self->last_emit_end;
  if (emitter.full) {
    return iree_ok_status();
  }

  // All emissions succeeded. Reset pattern matcher state for fresh input.
  if (self->is_literal) {
    self->literal_match_position = 0;
  } else {
    iree_tokenizer_regex_exec_initialize(&self->regex_state,
                                         &segmenter->regex.dfa);
  }

  // Reset position tracking to 0. After flush, the tokenizer compacts the
  // buffer and refills from position 0, so our next process() call will
  // receive a fresh chunk at offset 0.
  self->bytes_processed = 0;
  self->last_emit_end = 0;

  return iree_ok_status();
}

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_split_vtable = {
        .destroy = iree_tokenizer_segmenter_split_destroy,
        .state_initialize = iree_tokenizer_segmenter_split_state_initialize,
        .state_deinitialize = iree_tokenizer_segmenter_split_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_split_state_process,
        .state_finalize = iree_tokenizer_segmenter_split_state_finalize,
        .state_has_pending = iree_tokenizer_segmenter_split_state_has_pending,
        .state_flush = iree_tokenizer_segmenter_split_state_flush,
};
