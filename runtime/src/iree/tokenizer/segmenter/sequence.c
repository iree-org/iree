// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/sequence.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Sequence Segmenter Implementation
//===----------------------------------------------------------------------===//

// Sequence segmenter configuration.
typedef struct iree_tokenizer_segmenter_sequence_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
  iree_host_size_t child_count;
  // Children stored inline (MAX_DEPTH is small and fixed).
  iree_tokenizer_segmenter_t*
      children[IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH];
  // Child state offsets computed at allocation time for O(1) access.
  // These are byte offsets from the start of the state struct.
  iree_host_size_t
      child_state_offsets[IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH];
} iree_tokenizer_segmenter_sequence_t;

// Per-level state for the iterative pipeline.
//
// Each level tracks processing of one parent segment by child[level_index+1].
// The child's DFA state persists in the child state storage across calls,
// so resuming after output fills is O(1) rather than O(N).
typedef struct iree_tokenizer_segmenter_sequence_level_t {
  // Byte range of the parent segment, normalized to start at 0.
  // parent_segment.end equals the parent segment's byte length.
  iree_tokenizer_segment_t parent_segment;
  // True when this level has a parent segment to process.
  bool has_parent_segment;
  // True when the child returned consumed=0, indicating it needs finalize.
  bool needs_finalize;
  // True when finalize() has been called at least once for this parent segment.
  bool finalize_started;
  // Byte offset from the original input's start to the parent segment's start.
  // Added to child output offsets to convert to input-relative coordinates.
  iree_host_size_t offset_adjustment;
  // Bytes consumed by the child within the parent segment.
  iree_host_size_t child_consumed;
} iree_tokenizer_segmenter_sequence_level_t;

// Streaming state for Sequence segmenter.
//
// Uses an iterative demand-driven pipeline where each byte of input is
// processed by each child exactly once, regardless of output buffer size.
// When output fills, child DFA states persist and the next call continues from
// exactly where the previous call stopped.
//
// The pipeline has `child_count - 1` levels. Level i tracks child[i+1]'s
// processing of segments produced by child[i]. Child[0] is managed directly
// by state_process and state_finalize (it maintains true streaming state across
// input chunks), while children[1..n-1] see each parent segment as a complete
// mini-stream (process + finalize per parent segment).
typedef struct iree_tokenizer_segmenter_sequence_state_t {
  iree_tokenizer_segmenter_state_t base;

  // True when child[0] has produced a segment that is being expanded through
  // the pipeline. While true, state_process does not call child[0] for more
  // segments. Cleared once all levels exhaust (expansion complete).
  bool have_current_segment;

  // True once child[0].finalize() has returned all segments.
  bool finalize_child0_complete;

  // Bytes child[0] consumed to produce the current in-progress segment.
  // Committed as out_consumed only after the entire expansion completes.
  iree_host_size_t current_consumed;

  // Cumulative byte offset into remaining_input that finalize() has processed.
  iree_host_size_t finalize_input_offset;

  // Pipeline levels: levels[i] drives child[i+1].
  // Active level count = child_count - 1.
  iree_tokenizer_segmenter_sequence_level_t
      levels[IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH - 1];

  // Child states follow contiguously after this struct.
  // Access via: (uint8_t*)state + segmenter->child_state_offsets[i]
} iree_tokenizer_segmenter_sequence_state_t;

// Result from advancing a single pipeline level.
typedef enum {
  // Child produced a segment (written to out_segment).
  IREE_TOKENIZER_SEQUENCE_ADVANCE_GOT_SEGMENT,
  // Child consumed input bytes but didn't produce a segment yet.
  IREE_TOKENIZER_SEQUENCE_ADVANCE_CONSUMED,
  // Level's parent segment is fully processed and finalized.
  IREE_TOKENIZER_SEQUENCE_ADVANCE_EXHAUSTED,
} iree_tokenizer_sequence_advance_result_t;

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_sequence_vtable;

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

// Calculates the state size and populates child state offsets.
// Must be called during allocate() to set up the segmenter's
// child_state_offsets before state_size is cached in the base struct.
static iree_status_t iree_tokenizer_segmenter_sequence_calculate_state_layout(
    iree_tokenizer_segmenter_sequence_t* segmenter,
    iree_host_size_t* out_total_size) {
  // Build field descriptors for each child's state.
  // Since IREE_STRUCT_LAYOUT uses a compile-time array, we manually compute
  // the layout using iree_struct_layout_calculate for dynamic child counts.
  iree_struct_field_t fields[IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH];
  for (iree_host_size_t i = 0; i < segmenter->child_count; ++i) {
    iree_host_size_t child_state_size =
        iree_tokenizer_segmenter_state_size(segmenter->children[i]);
    fields[i] = (iree_struct_field_t){
        .count = {child_state_size, 1},
        .element_size = 1,  // Raw bytes.
        .alignment = iree_alignof(iree_max_align_t),
        .out_offset = &segmenter->child_state_offsets[i],
    };
  }
  return iree_struct_layout_calculate(
      sizeof(iree_tokenizer_segmenter_sequence_state_t), fields,
      segmenter->child_count, out_total_size);
}

iree_status_t iree_tokenizer_segmenter_sequence_allocate(
    iree_tokenizer_segmenter_t* const* children, iree_host_size_t child_count,
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  // Validate child count (must be at least 2, tokenizer uses singles directly).
  if (child_count < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sequence child count %" PRIhsz
        " is less than minimum 2; use single segmenters directly",
        child_count);
  }
  if (child_count > IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sequence child count %" PRIhsz " exceeds maximum %d", child_count,
        IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH);
  }

  // Validate all children are non-NULL.
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    if (!children[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "sequence child %" PRIhsz " is NULL", i);
    }
  }

  // Allocate segmenter struct.
  iree_tokenizer_segmenter_sequence_t* segmenter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter));

  segmenter->allocator = allocator;
  segmenter->child_count = child_count;

  // Copy child pointers (sequence takes ownership).
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    segmenter->children[i] = children[i];
  }

  // Calculate state size and populate child state offsets.
  iree_host_size_t total_state_size = 0;
  iree_status_t status =
      iree_tokenizer_segmenter_sequence_calculate_state_layout(
          segmenter, &total_state_size);

  // Initialize base with vtable and computed state size.
  if (iree_status_is_ok(status)) {
    iree_tokenizer_segmenter_initialize(
        &segmenter->base, &iree_tokenizer_segmenter_sequence_vtable,
        total_state_size);
    *out_segmenter = &segmenter->base;
  } else {
    iree_allocator_free(allocator, segmenter);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_tokenizer_segmenter_sequence_destroy(
    iree_tokenizer_segmenter_t* base_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_sequence_t* segmenter =
      (iree_tokenizer_segmenter_sequence_t*)base_segmenter;
  iree_allocator_t allocator = segmenter->allocator;

  // Free all child segmenters (sequence takes ownership).
  for (iree_host_size_t i = 0; i < segmenter->child_count; ++i) {
    iree_tokenizer_segmenter_free(segmenter->children[i]);
  }

  iree_allocator_free(allocator, segmenter);
  IREE_TRACE_ZONE_END(z0);
}

// Gets the child state at index |i|.
static inline iree_tokenizer_segmenter_state_t*
iree_tokenizer_segmenter_sequence_get_child_state(
    iree_tokenizer_segmenter_sequence_state_t* state,
    const iree_tokenizer_segmenter_sequence_t* segmenter, iree_host_size_t i) {
  return (iree_tokenizer_segmenter_state_t*)((uint8_t*)state +
                                             segmenter->child_state_offsets[i]);
}

static iree_status_t iree_tokenizer_segmenter_sequence_state_initialize(
    const iree_tokenizer_segmenter_t* base_segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_segmenter_sequence_t* segmenter =
      (const iree_tokenizer_segmenter_sequence_t*)base_segmenter;
  iree_tokenizer_segmenter_sequence_state_t* state =
      (iree_tokenizer_segmenter_sequence_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.segmenter = base_segmenter;
  state->have_current_segment = false;

  // Initialize each child state in the contiguous storage.
  for (iree_host_size_t i = 0; i < segmenter->child_count; ++i) {
    void* child_storage = (uint8_t*)storage + segmenter->child_state_offsets[i];
    iree_tokenizer_segmenter_state_t* child_state = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tokenizer_segmenter_state_initialize(
                segmenter->children[i], child_storage, &child_state));
  }

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_sequence_state_deinitialize(
    iree_tokenizer_segmenter_state_t* base_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_sequence_state_t* state =
      (iree_tokenizer_segmenter_sequence_state_t*)base_state;
  const iree_tokenizer_segmenter_sequence_t* segmenter =
      (const iree_tokenizer_segmenter_sequence_t*)base_state->segmenter;
  // Deinitialize child states in reverse order.
  for (iree_host_size_t i = segmenter->child_count; i > 0; --i) {
    iree_tokenizer_segmenter_state_t* child_state =
        iree_tokenizer_segmenter_sequence_get_child_state(state, segmenter,
                                                          i - 1);
    iree_tokenizer_segmenter_state_deinitialize(child_state);
  }
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Pipeline Helpers
//===----------------------------------------------------------------------===//

// Reinitializes a child's state in-place. Called when a level receives a new
// parent segment, since the child needs fresh DFA state for the new text
// region.
static iree_status_t iree_tokenizer_segmenter_sequence_reinit_child_state(
    iree_tokenizer_segmenter_sequence_state_t* state,
    const iree_tokenizer_segmenter_sequence_t* segmenter,
    iree_host_size_t child_index) {
  iree_tokenizer_segmenter_state_t* child_state =
      iree_tokenizer_segmenter_sequence_get_child_state(state, segmenter,
                                                        child_index);
  iree_tokenizer_segmenter_state_deinitialize(child_state);
  return iree_tokenizer_segmenter_state_initialize(
      segmenter->children[child_index],
      (uint8_t*)state + segmenter->child_state_offsets[child_index],
      &child_state);
}

// Configures a level with a new parent segment to process.
// |parent_segment| has input-relative coordinates. The level normalizes it
// to {0, length} and stores the original start as offset_adjustment.
static void iree_tokenizer_segmenter_sequence_setup_level_parent(
    iree_tokenizer_segmenter_sequence_level_t* level,
    iree_tokenizer_segment_t parent_segment) {
  level->parent_segment.start = 0;
  level->parent_segment.end = parent_segment.end - parent_segment.start;
  level->has_parent_segment = true;
  level->offset_adjustment = parent_segment.start;
  level->child_consumed = 0;
  level->needs_finalize = false;
  level->finalize_started = false;
}

// Tries to produce one segment from a pipeline level.
//
// Calls the level's child (child[level_index+1]) with the remaining parent
// segment text. The child's DFA state persists across calls, so this is
// O(new bytes) per call, not O(total bytes).
//
// The |input| parameter provides the text that level offsets reference into.
// |out_segment| receives input-relative coordinates when the result is
// GOT_SEGMENT.
static iree_status_t iree_tokenizer_segmenter_sequence_advance_child(
    iree_tokenizer_segmenter_sequence_state_t* state,
    const iree_tokenizer_segmenter_sequence_t* segmenter,
    iree_host_size_t level_index, iree_string_view_t input,
    iree_tokenizer_segment_t* out_segment,
    iree_tokenizer_sequence_advance_result_t* out_result) {
  iree_tokenizer_segmenter_sequence_level_t* level =
      &state->levels[level_index];
  iree_tokenizer_segmenter_state_t* child_state =
      iree_tokenizer_segmenter_sequence_get_child_state(state, segmenter,
                                                        level_index + 1);

  // Process path: feed remaining parent text to child.
  if (!level->needs_finalize) {
    iree_host_size_t remaining_length =
        level->parent_segment.end - level->child_consumed;
    iree_string_view_t remaining = iree_make_string_view(
        input.data + level->offset_adjustment + level->child_consumed,
        remaining_length);

    iree_tokenizer_segment_t segment;
    iree_tokenizer_segment_output_t segment_output =
        iree_tokenizer_make_segment_output(&segment, 1);
    iree_host_size_t consumed = 0;
    iree_host_size_t count = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_process(
        child_state, remaining, segment_output, &consumed, &count));

    // Save input offset before advancing child_consumed. The segment's
    // coordinates are relative to |remaining| which starts at this offset.
    iree_host_size_t input_offset =
        level->offset_adjustment + level->child_consumed;
    level->child_consumed += consumed;

    if (count > 0) {
      out_segment->start = segment.start + input_offset;
      out_segment->end = segment.end + input_offset;
      *out_result = IREE_TOKENIZER_SEQUENCE_ADVANCE_GOT_SEGMENT;
      return iree_ok_status();
    }

    if (consumed == 0) {
      // Child can't consume more without finalize.
      level->needs_finalize = true;
      // Fall through to finalize path.
    } else {
      // Child consumed bytes but didn't produce a segment yet (buffering).
      *out_result = IREE_TOKENIZER_SEQUENCE_ADVANCE_CONSUMED;
      return iree_ok_status();
    }
  }

  // Finalize path: drain remaining output from the child.
  iree_string_view_t remaining = iree_make_string_view(
      input.data + level->offset_adjustment + level->child_consumed,
      level->parent_segment.end - level->child_consumed);

  if (!level->finalize_started ||
      iree_tokenizer_segmenter_state_has_pending(child_state)) {
    iree_tokenizer_segment_t segment;
    iree_tokenizer_segment_output_t segment_output =
        iree_tokenizer_make_segment_output(&segment, 1);
    iree_host_size_t count = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_finalize(
        child_state, remaining, segment_output, &count));
    level->finalize_started = true;

    if (count > 0) {
      iree_host_size_t input_offset =
          level->offset_adjustment + level->child_consumed;
      out_segment->start = segment.start + input_offset;
      out_segment->end = segment.end + input_offset;
      *out_result = IREE_TOKENIZER_SEQUENCE_ADVANCE_GOT_SEGMENT;
      return iree_ok_status();
    }
  }

  *out_result = IREE_TOKENIZER_SEQUENCE_ADVANCE_EXHAUSTED;
  return iree_ok_status();
}

// Produces one final output segment by iteratively walking the pipeline levels.
//
// Starting from |final_level|, walks upward to find a level with a parent
// segment, advances that level's child, and feeds any produced segment to the
// next deeper level. When |final_level| produces a segment, it becomes the
// output.
//
// |final_level| is typically child_count - 2 (the deepest level).
//
// When all levels are exhausted, sets |out_exhausted| to true, indicating that
// the caller needs to provide a new segment to continue.
//
// The |input| parameter provides the text that all level offsets reference
// into.
static iree_status_t iree_tokenizer_segmenter_sequence_pull_next_output_segment(
    iree_tokenizer_segmenter_sequence_state_t* state,
    const iree_tokenizer_segmenter_sequence_t* segmenter,
    iree_string_view_t input, iree_tokenizer_segment_t* out_segment,
    bool* out_exhausted, iree_host_size_t final_level) {
  iree_host_size_t level = final_level;

  for (;;) {
    // Walk upward to find a level with a parent segment.
    while (!state->levels[level].has_parent_segment) {
      if (level == 0) {
        *out_exhausted = true;
        return iree_ok_status();
      }
      --level;
    }

    // Advance this level's child.
    iree_tokenizer_segment_t segment;
    iree_tokenizer_sequence_advance_result_t result;
    IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_sequence_advance_child(
        state, segmenter, level, input, &segment, &result));

    switch (result) {
      case IREE_TOKENIZER_SEQUENCE_ADVANCE_GOT_SEGMENT:
        if (level == final_level) {
          // Target level produced a segment: this is a final output.
          *out_segment = segment;
          *out_exhausted = false;
          return iree_ok_status();
        }
        // Feed segment to the next deeper level.
        iree_tokenizer_segmenter_sequence_setup_level_parent(
            &state->levels[level + 1], segment);
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_segmenter_sequence_reinit_child_state(
                state, segmenter, level + 2));
        level = final_level;
        continue;

      case IREE_TOKENIZER_SEQUENCE_ADVANCE_CONSUMED:
        // Child consumed bytes but no segment yet. Try same level again.
        continue;

      case IREE_TOKENIZER_SEQUENCE_ADVANCE_EXHAUSTED:
        // Level's parent segment fully processed.
        state->levels[level].has_parent_segment = false;
        if (level == 0) {
          *out_exhausted = true;
          return iree_ok_status();
        }
        --level;
        continue;
    }
  }
}

//===----------------------------------------------------------------------===//
// Processing
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_segmenter_sequence_state_process(
    iree_tokenizer_segmenter_state_t* base_state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_sequence_state_t* state =
      (iree_tokenizer_segmenter_sequence_state_t*)base_state;
  const iree_tokenizer_segmenter_sequence_t* segmenter =
      (const iree_tokenizer_segmenter_sequence_t*)base_state->segmenter;

  *out_consumed = 0;
  *out_segment_count = 0;

  if (input.size == 0 || output.capacity == 0) {
    return iree_ok_status();
  }

  iree_host_size_t output_count = 0;
  iree_host_size_t total_consumed = 0;

  iree_host_size_t final_level = segmenter->child_count - 2;

  // Resume a pending expansion from a previous call that filled the output.
  // The pipeline levels and child DFA states are preserved, so this continues
  // exactly where the previous call stopped.
  if (state->have_current_segment) {
    bool exhausted = false;
    while (output_count < output.capacity) {
      iree_tokenizer_segment_t segment;
      IREE_RETURN_IF_ERROR(
          iree_tokenizer_segmenter_sequence_pull_next_output_segment(
              state, segmenter, input, &segment, &exhausted, final_level));
      if (exhausted) break;
      output.values[output_count++] = segment;
    }

    if (exhausted) {
      // Expansion complete. Commit the consumption from child[0].
      total_consumed = state->current_consumed;
      state->have_current_segment = false;
    } else {
      // Output filled again mid-expansion. Pipeline state is preserved.
      *out_consumed = 0;
      *out_segment_count = output_count;
      return iree_ok_status();
    }
  }

  // Get segments from child[0] and expand each through the pipeline.
  iree_tokenizer_segmenter_state_t* child0_state =
      iree_tokenizer_segmenter_sequence_get_child_state(state, segmenter, 0);

  iree_string_view_t remaining = iree_make_string_view(
      input.data + total_consumed, input.size - total_consumed);

  while (remaining.size > 0 && output_count < output.capacity) {
    iree_tokenizer_segment_t child0_segment;
    iree_tokenizer_segment_output_t child0_output =
        iree_tokenizer_make_segment_output(&child0_segment, 1);

    iree_host_size_t consumed = 0;
    iree_host_size_t child0_count = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_process(
        child0_state, remaining, child0_output, &consumed, &child0_count));

    if (child0_count == 0) {
      if (consumed == 0 && remaining.size > 0 &&
          !iree_tokenizer_segmenter_state_has_pending(child0_state)) {
        // child[0] found no matches in the visible text (e.g., Numbers split
        // on English text without digits). child[0]'s output is identity: one
        // pass-through segment covering all visible text. Feed this through the
        // normal pipeline (child[1] → child[2] → ...) to preserve correct
        // child ordering. This avoids the ring buffer stall that would result
        // from returning consumed=0 when child[0] genuinely has nothing to
        // match.
        //
        // Before entering pass-through, probe the final child to verify the
        // pipeline can make progress on this text. If the final child (the most
        // general pattern matcher, e.g., MainRegex) also returns consumed=0
        // with no pending state, the text is too small or incomplete for any
        // child to process. Return consumed=0 to let the caller provide more
        // data or call finalize(). Without this probe, the pipeline's internal
        // finalize path would force partial text through (e.g., splitting a
        // partial UTF-8 sequence into separate segments).
        // Probe the final child (the most general pattern matcher, e.g.,
        // MainRegex) to find the total bytes it can consume. We call
        // process() in a loop, discarding its output segments, to accumulate
        // the total consumable byte count. The final child's DFA stops at
        // natural word/match boundaries, so the pass-through segment will end
        // cleanly without splitting words at arbitrary buffer boundaries.
        //
        // This loop is critical for avoiding O(N^2) behavior: without it,
        // each pass-through would consume only one word (~25 bytes), and the
        // outer loop would reinitialize + rescan child[0] over the full
        // remaining buffer on every iteration.
        iree_host_size_t final_child_index = segmenter->child_count - 1;
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_segmenter_sequence_reinit_child_state(
                state, segmenter, final_child_index));
        iree_tokenizer_segmenter_state_t* final_child_state =
            iree_tokenizer_segmenter_sequence_get_child_state(
                state, segmenter, final_child_index);
        iree_host_size_t total_probe_consumed = 0;
        {
          iree_tokenizer_segment_t probe_segments[64];
          iree_string_view_t probe_remaining = remaining;
          while (probe_remaining.size > 0) {
            iree_host_size_t probe_consumed = 0;
            iree_host_size_t probe_count = 0;
            IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_process(
                final_child_state, probe_remaining,
                iree_tokenizer_make_segment_output(probe_segments, 64),
                &probe_consumed, &probe_count));
            if (probe_consumed == 0) break;
            total_probe_consumed += probe_consumed;
            probe_remaining.data += probe_consumed;
            probe_remaining.size -= probe_consumed;
          }
        }
        if (total_probe_consumed == 0) {
          // Final child can't consume any bytes either. The text is too small
          // or incomplete for any child to process — return consumed=0 to let
          // the caller provide more data or call finalize().
          total_consumed += consumed;
          break;
        }

        iree_tokenizer_segment_t passthrough_segment = {
            .start = (iree_host_size_t)(remaining.data - input.data),
            .end = (iree_host_size_t)(remaining.data - input.data) +
                   total_probe_consumed,
        };
        iree_host_size_t passthrough_size = total_probe_consumed;

        iree_tokenizer_segmenter_sequence_setup_level_parent(
            &state->levels[0], passthrough_segment);
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_segmenter_sequence_reinit_child_state(state,
                                                                 segmenter, 1));

        bool expansion_complete = false;
        while (output_count < output.capacity) {
          iree_tokenizer_segment_t segment;
          bool exhausted = false;
          IREE_RETURN_IF_ERROR(
              iree_tokenizer_segmenter_sequence_pull_next_output_segment(
                  state, segmenter, input, &segment, &exhausted, final_level));
          if (exhausted) {
            expansion_complete = true;
            break;
          }
          output.values[output_count++] = segment;
        }

        if (expansion_complete) {
          total_consumed += passthrough_size;
          remaining.data += passthrough_size;
          remaining.size -= passthrough_size;
          continue;
        } else {
          // Output filled mid-expansion. Save state for resumption.
          state->have_current_segment = true;
          state->current_consumed = total_consumed + passthrough_size;
          *out_consumed = 0;
          *out_segment_count = output_count;
          return iree_ok_status();
        }
      }
      // child[0] is buffering or has pending state.
      total_consumed += consumed;
      break;
    }

    // Got a segment from child[0]. Adjust to input-relative coordinates.
    iree_tokenizer_segment_t adjusted_segment = {
        .start = child0_segment.start + (remaining.data - input.data),
        .end = child0_segment.end + (remaining.data - input.data),
    };

    // Set up the pipeline with this segment and expand through all children.
    iree_tokenizer_segmenter_sequence_setup_level_parent(&state->levels[0],
                                                         adjusted_segment);
    IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_sequence_reinit_child_state(
        state, segmenter, 1));

    bool expansion_complete = false;
    while (output_count < output.capacity) {
      iree_tokenizer_segment_t segment;
      bool exhausted = false;
      IREE_RETURN_IF_ERROR(
          iree_tokenizer_segmenter_sequence_pull_next_output_segment(
              state, segmenter, input, &segment, &exhausted, final_level));
      if (exhausted) {
        expansion_complete = true;
        break;
      }
      output.values[output_count++] = segment;
    }

    if (expansion_complete) {
      // Fully expanded. Advance past what child[0] consumed.
      total_consumed += consumed;
      remaining.data += consumed;
      remaining.size -= consumed;
    } else {
      // Output filled mid-expansion. Save state for resumption.
      // Pipeline levels and child DFA states are already preserved.
      state->have_current_segment = true;
      state->current_consumed = total_consumed + consumed;
      *out_consumed = 0;  // Don't consume until fully expanded.
      *out_segment_count = output_count;
      return iree_ok_status();
    }
  }

  *out_consumed = total_consumed;
  *out_segment_count = output_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_segmenter_sequence_state_finalize(
    iree_tokenizer_segmenter_state_t* base_state,
    iree_string_view_t remaining_input, iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_sequence_state_t* state =
      (iree_tokenizer_segmenter_sequence_state_t*)base_state;
  const iree_tokenizer_segmenter_sequence_t* segmenter =
      (const iree_tokenizer_segmenter_sequence_t*)base_state->segmenter;

  iree_host_size_t output_count = 0;

  // Apply cumulative offset from previous finalize() calls. When we store
  // pipeline state, level offsets are relative to this adjusted position.
  // We save base_offset so we can adjust output segments back to the caller's
  // coordinate system before returning.
  iree_host_size_t base_offset = state->finalize_input_offset;
  remaining_input.data += base_offset;
  remaining_input.size -= base_offset;

  iree_host_size_t final_level = segmenter->child_count - 2;

  // Resume a pending pipeline expansion from process() or a prior finalize().
  if (state->have_current_segment) {
    bool exhausted = false;
    while (output_count < output.capacity) {
      iree_tokenizer_segment_t segment;
      IREE_RETURN_IF_ERROR(
          iree_tokenizer_segmenter_sequence_pull_next_output_segment(
              state, segmenter, remaining_input, &segment, &exhausted,
              final_level));
      if (exhausted) break;
      output.values[output_count++] = segment;
    }

    if (exhausted) {
      // Expansion complete. Advance past what child[0] consumed during
      // process() but didn't commit (because we returned consumed=0).
      iree_host_size_t consumed_in_process = state->current_consumed;
      remaining_input.data += consumed_in_process;
      remaining_input.size -= consumed_in_process;
      state->finalize_input_offset += consumed_in_process;
      base_offset += consumed_in_process;

      state->have_current_segment = false;
      state->current_consumed = 0;

      // If child[0] has no more pending segments, mark it complete.
      iree_tokenizer_segmenter_state_t* child0_state =
          iree_tokenizer_segmenter_sequence_get_child_state(state, segmenter,
                                                            0);
      if (!iree_tokenizer_segmenter_state_has_pending(child0_state)) {
        state->finalize_child0_complete = true;
      }
    } else {
      // Output filled mid-expansion. Pipeline state is preserved.
      for (iree_host_size_t i = 0; i < output_count; ++i) {
        output.values[i].start += base_offset;
        output.values[i].end += base_offset;
      }
      *out_segment_count = output_count;
      return iree_ok_status();
    }
  }

  // Skip child[0].finalize() if we've already drained all its segments.
  if (state->finalize_child0_complete) {
    for (iree_host_size_t i = 0; i < output_count; ++i) {
      output.values[i].start += base_offset;
      output.values[i].end += base_offset;
    }
    *out_segment_count = output_count;
    return iree_ok_status();
  }

  // Get segments from child[0].finalize() one at a time and expand each
  // through the pipeline.
  iree_tokenizer_segmenter_state_t* child0_state =
      iree_tokenizer_segmenter_sequence_get_child_state(state, segmenter, 0);

  iree_tokenizer_segment_t child0_segment;
  iree_tokenizer_segment_output_t child0_output =
      iree_tokenizer_make_segment_output(&child0_segment, 1);

  // Loop while child[0] has pending segments or we haven't called finalize yet.
  bool need_first_call =
      !iree_tokenizer_segmenter_state_has_pending(child0_state);

  while (iree_tokenizer_segmenter_state_has_pending(child0_state) ||
         need_first_call) {
    need_first_call = false;

    iree_host_size_t child0_count = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_finalize(
        child0_state, remaining_input, child0_output, &child0_count));

    if (child0_count == 0) {
      state->finalize_child0_complete = true;
      break;
    }

    // Set up pipeline with child[0]'s finalize segment and expand.
    iree_tokenizer_segmenter_sequence_setup_level_parent(&state->levels[0],
                                                         child0_segment);
    IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_sequence_reinit_child_state(
        state, segmenter, 1));

    bool expansion_complete = false;
    while (output_count < output.capacity) {
      iree_tokenizer_segment_t segment;
      bool exhausted = false;
      IREE_RETURN_IF_ERROR(
          iree_tokenizer_segmenter_sequence_pull_next_output_segment(
              state, segmenter, remaining_input, &segment, &exhausted,
              final_level));
      if (exhausted) {
        expansion_complete = true;
        break;
      }
      output.values[output_count++] = segment;
    }

    if (!expansion_complete) {
      // Output filled mid-expansion. Save state for next finalize() call.
      state->have_current_segment = true;
      // current_consumed = 0 because finalize segments are already relative
      // to the adjusted remaining_input (no additional offset needed).
      state->current_consumed = 0;
      for (iree_host_size_t i = 0; i < output_count; ++i) {
        output.values[i].start += base_offset;
        output.values[i].end += base_offset;
      }
      *out_segment_count = output_count;
      return iree_ok_status();
    }
  }

  // Adjust all output segments to the caller's coordinate system.
  for (iree_host_size_t i = 0; i < output_count; ++i) {
    output.values[i].start += base_offset;
    output.values[i].end += base_offset;
  }
  *out_segment_count = output_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Has Pending
//===----------------------------------------------------------------------===//

static bool iree_tokenizer_segmenter_sequence_state_has_pending(
    const iree_tokenizer_segmenter_state_t* base_state) {
  const iree_tokenizer_segmenter_sequence_state_t* state =
      (const iree_tokenizer_segmenter_sequence_state_t*)base_state;
  const iree_tokenizer_segmenter_sequence_t* segmenter =
      (const iree_tokenizer_segmenter_sequence_t*)base_state->segmenter;

  // Check if we have a pipeline expansion in progress.
  if (state->have_current_segment) {
    return true;
  }

  // Check if any pipeline level has a parent segment being processed.
  iree_host_size_t number_of_levels = segmenter->child_count - 1;
  for (iree_host_size_t i = 0; i < number_of_levels; ++i) {
    if (state->levels[i].has_parent_segment) {
      return true;
    }
  }

  // Check if any child has pending data.
  for (iree_host_size_t i = 0; i < segmenter->child_count; ++i) {
    const iree_tokenizer_segmenter_state_t* child_state =
        (const iree_tokenizer_segmenter_state_t*)((const uint8_t*)state +
                                                  segmenter
                                                      ->child_state_offsets[i]);
    if (iree_tokenizer_segmenter_state_has_pending(child_state)) {
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_sequence_vtable = {
        .destroy = iree_tokenizer_segmenter_sequence_destroy,
        .state_initialize = iree_tokenizer_segmenter_sequence_state_initialize,
        .state_deinitialize =
            iree_tokenizer_segmenter_sequence_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_sequence_state_process,
        .state_finalize = iree_tokenizer_segmenter_sequence_state_finalize,
        .state_has_pending =
            iree_tokenizer_segmenter_sequence_state_has_pending,
};
