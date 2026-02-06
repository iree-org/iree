// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE encoding state machine.
//
// This file implements the resumable state machine for segment-by-segment
// tokenization. The state machine handles:
// - Fast-path whole-segment matching (single-token optimization)
// - O(n) backtracking for short segments
// - O(n log L) sliding window for long segments
// - Streaming partial segment processing

#include "iree/tokenizer/model/bpe_internal.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"

//===----------------------------------------------------------------------===//
// BPE Segment Encoding
//===----------------------------------------------------------------------===//

// Encodes a single segment using a resumable state machine.
//
// State machine phases (see iree_tokenizer_bpe_phase_t):
//   SEGMENT_START: Initial state. Try fast-path trie match, else route.
//   FAST_PATH_PENDING: Fast-path token matched but output was full.
//   BACKTRACK_EMIT: Emitting tokens from the backtrack stack.
//   BYTE_LOOP: Processing input bytes, building window, emitting frozen tokens.
//   FLUSH: All bytes processed, emitting remaining window tokens.
//
// Transitions:
//   SEGMENT_START -> BYTE_LOOP (normal path, segment > backtrack threshold)
//   SEGMENT_START -> [complete] (fast-path: whole segment = one token)
//   SEGMENT_START -> FAST_PATH_PENDING (fast-path match, output full)
//   SEGMENT_START -> BACKTRACK_EMIT (short segment, backtracking path)
//   FAST_PATH_PENDING -> [complete] (emit pending token)
//   BACKTRACK_EMIT -> [complete] (all backtrack tokens emitted)
//   BACKTRACK_EMIT -> BACKTRACK_EMIT (output full, resume next call)
//   BYTE_LOOP -> FLUSH (all bytes processed)
//   BYTE_LOOP -> BYTE_LOOP (output full mid-loop, resume next call)
//   FLUSH -> [complete] (all window tokens emitted)
//   FLUSH -> FLUSH (output full, resume next call)
//
// Returns true in |out_segment_complete| when the segment is fully processed.
static iree_status_t iree_tokenizer_bpe_encode_segment(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment, iree_tokenizer_token_id_t* out_tokens,
    iree_tokenizer_offset_t* out_offsets, iree_host_size_t max_tokens,
    iree_host_size_t segment_base_offset, bool is_partial,
    iree_host_size_t* out_token_count, bool* out_segment_complete) {
  *out_token_count = 0;
  *out_segment_complete = false;

  if (segment.size == 0) {
    *out_segment_complete = true;
    return iree_ok_status();
  }

  // Track original segment size for offset clamping. When end_of_word_suffix
  // is present, the suffix is part of the vocabulary but not the original
  // input, so byte offsets must be clamped to original_segment_size.
  iree_host_size_t original_segment_size = segment.size;

  iree_tokenizer_bpe_output_cursor_t cursor =
      iree_tokenizer_bpe_output_cursor_make(out_tokens, out_offsets,
                                            segment_base_offset, max_tokens);

  // Track segment state for finalize and offset clamping.
  // Updated on each call since reclaim shifts the segment in the ring buffer.
  if (is_partial) {
    state->segment.base_offset = segment_base_offset;
  }
  state->segment.original_size = original_segment_size;

  // FAST_PATH_PENDING: Resume from previous call that matched whole segment
  // but couldn't emit because output was full.
  if (state->phase == IREE_TOKENIZER_BPE_PHASE_FAST_PATH_PENDING) {
    if (!iree_tokenizer_bpe_emit_and_track(
            state, &cursor, state->fast_path_pending_token_id, 0,
            (uint32_t)state->segment.original_size)) {
      // Still can't emit. Return with no progress.
      *out_token_count = 0;
      return iree_ok_status();
    }
    // Token emitted. Segment complete.
    state->fast_path_pending_token_id = -1;
    state->phase = IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;
    *out_token_count =
        iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
    *out_segment_complete = true;
    return iree_ok_status();
  }

  // SEGMENT_START: Beginning a new segment. Try fast-path first.
  if (state->phase == IREE_TOKENIZER_BPE_PHASE_SEGMENT_START) {
    // Partial segments must use BYTE_LOOP regardless of size. Fast-path and
    // backtracking assume complete segments and would emit non-frozen tokens.
    if (is_partial) {
      state->window.count = 0;
      state->window.start = 0;
      state->segment.byte_position = 0;
      iree_tokenizer_bpe_heap_reset(&state->heap);
      state->phase = IREE_TOKENIZER_BPE_PHASE_BYTE_LOOP;
    } else {
      // Word cache: skip trie/backtracking for previously-tokenized segments.
      if (iree_tokenizer_bpe_cache_lookup(model, state, segment, &cursor)) {
        *out_token_count =
            iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
        *out_segment_complete = true;
        return iree_ok_status();
      }

      // Suffix-aware whole-segment match: when end_of_word_suffix is set,
      // check if segment + suffix matches exactly as a vocabulary token.
      // This bypasses the normal BPE algorithm because:
      //   - The token exists in vocabulary (we find it in trie)
      //   - The suffix handling is explicit from tokenizer config
      //   - No BPE merging is needed when whole segment is a single token
      // Example: CLIP's "hello" segment matches "hello</w>" token.
      //
      // SKIP when IGNORE_MERGES: HuggingFace's ignore_merges does longest-match
      // vocab lookup on the BARE segment (without suffix). Suffix-aware lookup
      // would incorrectly find "hello</w>" when HuggingFace finds "hello".
      const bool ignore_merges = iree_all_bits_set(
          model->flags, IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES);
      if (model->end_of_word_suffix_length > 0 && !ignore_merges) {
        iree_string_view_t suffix = iree_make_string_view(
            model->end_of_word_suffix, model->end_of_word_suffix_length);
        int32_t suffixed_token_id = -1;
        iree_host_size_t suffixed_match_length = 0;
        iree_tokenizer_bpe_trie_longest_match_with_suffix(
            model, segment, suffix, &suffixed_token_id, &suffixed_match_length);

        if (suffixed_token_id >= 0 &&
            suffixed_match_length == segment.size + suffix.size) {
          if (!iree_tokenizer_bpe_emit_and_track(
                  state, &cursor, suffixed_token_id, 0,
                  (uint32_t)original_segment_size)) {
            state->fast_path_pending_token_id = suffixed_token_id;
            state->phase = IREE_TOKENIZER_BPE_PHASE_FAST_PATH_PENDING;
            *out_token_count = 0;
            return iree_ok_status();
          }
          *out_token_count =
              iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
          *out_segment_complete = true;
          return iree_ok_status();
        }
      }

      // Fast path: check if entire segment matches a single vocabulary token.
      // This is only valid when:
      //   1. segment.size == 1 (single byte, no merges can apply), OR
      //   2. ignore_merges is set (HuggingFace BPE option to skip merge rules)
      // When ignore_merges is false (default), multi-byte segments must go
      // through the full BPE algorithm to respect merge priority order.
      const bool can_use_fast_path = segment.size == 1 || ignore_merges;

      if (can_use_fast_path) {
        int32_t whole_segment_token_id = -1;
        iree_host_size_t match_length = 0;
        iree_host_size_t expected_length = segment.size;

        // Use suffix-aware match for single-byte segments (segment.size == 1)
        // when end_of_word_suffix is set. For IGNORE_MERGES, always use bare
        // segment lookup since HuggingFace does longest-match without suffix.
        if (model->end_of_word_suffix_length > 0 && !ignore_merges) {
          iree_string_view_t suffix = iree_make_string_view(
              model->end_of_word_suffix, model->end_of_word_suffix_length);
          iree_tokenizer_bpe_trie_longest_match_with_suffix(
              model, segment, suffix, &whole_segment_token_id, &match_length);
          expected_length = segment.size + suffix.size;
        } else {
          // Use ByteLevel-aware lookup to handle byte-to-character mapping.
          iree_tokenizer_bpe_trie_longest_match_byte_level(
              model, segment, &whole_segment_token_id, &match_length);
        }

        if (whole_segment_token_id >= 0 && match_length == expected_length) {
          // Emit with original_segment_size for correct offset.
          if (!iree_tokenizer_bpe_emit_and_track(
                  state, &cursor, whole_segment_token_id, 0,
                  (uint32_t)original_segment_size)) {
            state->fast_path_pending_token_id = whole_segment_token_id;
            state->phase = IREE_TOKENIZER_BPE_PHASE_FAST_PATH_PENDING;
            *out_token_count = 0;
            return iree_ok_status();
          }
          *out_token_count =
              iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
          *out_segment_complete = true;
          return iree_ok_status();
        }
      }

      // Use backtracking for segments within threshold. This is the O(n)
      // path and handles 99.9%+ of real segments (word-level from
      // metaspace/whitespace/regex splitters are typically < 100 bytes).
      if (segment.size <= model->max_backtrack_segment_bytes) {
        iree_tokenizer_bpe_backtrack_encode(
            model, state, (const uint8_t*)segment.data, segment.size,
            model->end_of_word_suffix, model->end_of_word_suffix_length);
        // For models with end_of_word_suffix, apply suffix-aware
        // post-processing: replace last token with its suffixed version,
        // then recursively merge with previous tokens while merges exist.
        if (model->end_of_word_suffix_length > 0) {
          iree_tokenizer_bpe_apply_suffix_to_backtrack(model, state, segment);
        }
        state->phase = IREE_TOKENIZER_BPE_PHASE_BACKTRACK_EMIT;
        // Fall through to BACKTRACK_EMIT handler below.
      } else {
        // Segment exceeds backtracking threshold. Fall back to the O(n log L)
        // sliding window + min-heap algorithm.
        state->window.count = 0;
        state->window.start = 0;
        state->segment.byte_position = 0;
        iree_tokenizer_bpe_heap_reset(&state->heap);
        state->phase = IREE_TOKENIZER_BPE_PHASE_BYTE_LOOP;
      }
    }
  }

  // BACKTRACK_EMIT: Emit tokens from the backtrack stack. Reached either
  // from SEGMENT_START (first entry after backtrack_encode completes) or
  // on function re-entry when resuming after output-buffer-full.
  if (state->phase == IREE_TOKENIZER_BPE_PHASE_BACKTRACK_EMIT) {
    iree_tokenizer_bpe_backtrack_entry_t* stack =
        iree_tokenizer_bpe_state_backtrack_stack(state, model);
    while (state->backtrack.emit_index < state->backtrack.stack_count) {
      uint32_t start_byte = iree_tokenizer_bpe_backtrack_entry_start_byte(
          &stack[state->backtrack.emit_index]);
      uint32_t end_byte =
          (state->backtrack.emit_index + 1 < state->backtrack.stack_count)
              ? iree_tokenizer_bpe_backtrack_entry_start_byte(
                    &stack[state->backtrack.emit_index + 1])
              : (uint32_t)segment.size;
      if (!iree_tokenizer_bpe_emit_and_track(
              state, &cursor, stack[state->backtrack.emit_index].token_id,
              start_byte, end_byte)) {
        // Output full. Preserve emit_index for next call.
        *out_token_count =
            iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
        return iree_ok_status();
      }
      state->backtrack.emit_index++;
    }
    // All tokens emitted. Populate word cache for future lookups.
    iree_tokenizer_bpe_cache_populate(model, state, segment, stack,
                                      state->backtrack.stack_count);
    state->phase = IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;
    *out_token_count =
        iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
    *out_segment_complete = true;
    return iree_ok_status();
  }

  // BYTE_LOOP: Process input bytes one at a time, building up the window.
  // After adding each token, emit any tokens that are now "frozen"
  // (can't be affected by future input, per the frozen token theorem).
  //
  // Uses original segment (not effective_segment). The suffix is only used
  // for the fast-path whole-segment match. If that fails, we tokenize the
  // original segment without suffix.
  if (state->phase == IREE_TOKENIZER_BPE_PHASE_BYTE_LOOP) {
    for (iree_host_size_t byte_position = state->segment.byte_position;
         byte_position < segment.size; ++byte_position) {
      uint8_t input_byte = (uint8_t)segment.data[byte_position];

      // Look up single-byte token. For ByteLevel mode, this is the UTF-8
      // encoding of the ByteLevel codepoint (e.g., space -> "Ġ").
      int32_t token_id = model->byte_to_token[input_byte];
      iree_host_size_t token_byte_length = 1;
      if (token_id < 0) {
        // No direct single-byte token. Try a multi-byte trie match first:
        // some base vocabulary tokens (e.g., SentencePiece's ▁ = U+2581,
        // 3 bytes) span multiple raw bytes but are never produced by a merge.
        // The trie walk finds the longest such token; effective_rank filtering
        // inside backtrack_longest_match ensures we only accept tokens that
        // participate in BPE (not stray added tokens).
        iree_host_size_t trie_raw_length = 0;
        iree_tokenizer_bpe_backtrack_longest_match(
            model, (const uint8_t*)segment.data + byte_position,
            segment.size - byte_position, &token_id, &trie_raw_length);
        if (token_id >= 0) {
          token_byte_length = trie_raw_length;
        } else {
          // No trie match. Fall back to byte-fallback (<0xNN>) or UNK.
          // For FUSE_UNK, check the last token in the window (not yet emitted)
          // before falling back to last_emitted_token_id.
          int32_t previous_token =
              (state->window.count > 0)
                  ? iree_tokenizer_bpe_window_at(state, model,
                                                 state->window.count - 1)
                        ->token_id
                  : state->last_emitted_token_id;
          token_id = iree_tokenizer_bpe_handle_unknown_byte(model, input_byte,
                                                            previous_token);
          if (token_id < 0) {
            // Byte was fused with previous UNK. Skip to next byte.
            continue;
          }
        }
      }

      // Add token to the sliding window.
      iree_host_size_t token_end_byte = byte_position + token_byte_length;
      iree_tokenizer_bpe_window_token_t window_token = {
          token_id,
          (uint32_t)byte_position,
          (uint32_t)token_end_byte,
      };
      iree_tokenizer_bpe_window_push(state, model, window_token);

      // Check if the new token can merge with its left neighbor.
      if (state->window.count >= 2) {
        iree_tokenizer_bpe_maybe_add_merge(state, model,
                                           state->window.count - 2);
      }

      // Emit frozen tokens. The freeze check uses the last consumed byte
      // position (token_end_byte - 1) as the current frontier.
      if (!iree_tokenizer_bpe_emit_frozen_tokens(state, model,
                                                 token_end_byte - 1, &cursor)) {
        // Output full. Save next unprocessed byte for resumption.
        state->segment.byte_position = token_end_byte;
        *out_token_count =
            iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
        return iree_ok_status();
      }

      // Advance past remaining bytes of multi-byte token (loop does +1).
      byte_position = token_end_byte - 1;
    }

    if (is_partial) {
      // Streaming mode: more bytes will be appended. Stay in BYTE_LOOP and
      // preserve window/heap state for continuation. The caller will extend
      // the segment and re-enter.
      state->segment.byte_position = segment.size;
      *out_token_count =
          iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
      return iree_ok_status();
      // segment_complete stays false.
    }

    // All bytes processed. Transition to flush phase.
    state->segment.byte_position = 0;
    state->phase = IREE_TOKENIZER_BPE_PHASE_FLUSH;
  }

  // FLUSH: All input bytes processed. Apply remaining merges, then emit
  // all remaining tokens from the window.
  IREE_ASSERT(state->phase == IREE_TOKENIZER_BPE_PHASE_FLUSH);

  // Apply all remaining merges in rank order.
  iree_tokenizer_bpe_apply_pending_merges(state, model);

  // For models with end_of_word_suffix, apply suffix-aware post-processing:
  // replace last token with its suffixed version, then merge backwards.
  if (model->end_of_word_suffix_length > 0) {
    iree_tokenizer_bpe_apply_suffix_to_last_window_token(model, state, segment);
  }

  // Emit all remaining window tokens.
  // Since we use segment (not effective_segment) for BYTE_LOOP, offsets
  // are already within the original segment bounds.
  while (state->window.count > 0) {
    iree_tokenizer_bpe_window_token_t* front =
        iree_tokenizer_bpe_window_at(state, model, 0);
    if (!iree_tokenizer_bpe_emit_and_track(state, &cursor, front->token_id,
                                           front->start_byte,
                                           front->end_byte)) {
      // Output full. Window preserves remaining tokens for next call.
      *out_token_count =
          iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
      return iree_ok_status();
    }
    iree_tokenizer_bpe_window_pop_front(state, model);
  }

  // Segment complete. Reset state for next segment.
  state->phase = IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;
  *out_token_count =
      iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
  *out_segment_complete = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BPE State Encode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_bpe_state_encode(
    iree_tokenizer_model_state_t* state,
    iree_const_byte_span_t transform_buffer,
    iree_tokenizer_segment_list_t segments,
    iree_tokenizer_token_output_t output,
    iree_host_size_t* out_segments_consumed,
    iree_host_size_t* out_token_count) {
  iree_tokenizer_bpe_state_t* bpe_state = (iree_tokenizer_bpe_state_t*)state;
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)state->model;

  *out_segments_consumed = 0;
  *out_token_count = 0;

  iree_host_size_t total_tokens_written = 0;
  iree_host_size_t remaining_capacity = output.capacity;
  iree_tokenizer_token_id_t* current_token_output = output.token_ids;
  iree_tokenizer_offset_t* current_offset_output = output.token_offsets;

  // Process segments from the start of the provided list.
  // The caller is responsible for slicing the segment list when resuming
  // after a partial return - we always start from index 0.
  // Intra-segment state (byte_position, in_flush_phase, etc.) is preserved.
  for (iree_host_size_t i = 0; i < segments.count; ++i) {
    // Early exit when output buffer is full - no point processing more
    // segments.
    if (remaining_capacity == 0) break;

    const iree_tokenizer_segment_t* segment = &segments.values[i];

    if (segment->end > transform_buffer.data_length ||
        segment->start > segment->end) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "segment bounds out of range: [%" PRIhsz
                              ", %" PRIhsz ") in buffer of %" PRIhsz " bytes",
                              segment->start, segment->end,
                              transform_buffer.data_length);
    }

    iree_string_view_t segment_text = iree_make_string_view(
        (const char*)transform_buffer.data + segment->start,
        segment->end - segment->start);

    iree_host_size_t tokens_for_segment = 0;
    bool segment_complete = false;
    bool is_partial = segments.last_is_partial && (i == segments.count - 1);
    IREE_RETURN_IF_ERROR(iree_tokenizer_bpe_encode_segment(
        model, bpe_state, segment_text, current_token_output,
        current_offset_output, remaining_capacity, segment->start, is_partial,
        &tokens_for_segment, &segment_complete));

    total_tokens_written += tokens_for_segment;
    remaining_capacity -= tokens_for_segment;
    current_token_output += tokens_for_segment;
    if (current_offset_output) {
      current_offset_output += tokens_for_segment;
    }

    if (segment_complete) {
      (*out_segments_consumed)++;
    } else {
      // Output full mid-segment - stop processing.
      // State preserves position for next call.
      break;
    }
  }

  *out_token_count = total_tokens_written;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_bpe_state_finalize(
    iree_tokenizer_model_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* out_token_count) {
  iree_tokenizer_bpe_state_t* bpe_state = (iree_tokenizer_bpe_state_t*)state;
  *out_token_count = 0;

  // Flush remaining window tokens from two scenarios:
  //  - BYTE_LOOP: partial segment processing (streaming mode) never
  //  transitioned
  //    to FLUSH. Apply merges and emit.
  //  - FLUSH: output filled during a non-partial segment's FLUSH phase. The
  //    merges were already applied; just emit remaining window tokens.
  if (bpe_state->window.count == 0) {
    return iree_ok_status();
  }
  if (bpe_state->phase != IREE_TOKENIZER_BPE_PHASE_BYTE_LOOP &&
      bpe_state->phase != IREE_TOKENIZER_BPE_PHASE_FLUSH) {
    return iree_ok_status();
  }

  // Only now do we need the model (window access, merge application).
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)state->model;

  // Apply any remaining merges in the window before flushing.
  iree_tokenizer_bpe_apply_pending_merges(bpe_state, model);

  // Emit all window tokens using the stored segment base offset.
  iree_tokenizer_bpe_output_cursor_t cursor =
      iree_tokenizer_bpe_output_cursor_make(
          output.token_ids, output.token_offsets,
          bpe_state->segment.base_offset, output.capacity);
  while (bpe_state->window.count > 0) {
    iree_tokenizer_bpe_window_token_t* front =
        iree_tokenizer_bpe_window_at(bpe_state, model, 0);
    if (front->token_id < 0) {
      iree_tokenizer_bpe_window_pop_front(bpe_state, model);
      continue;
    }
    if (!iree_tokenizer_bpe_emit_and_track(bpe_state, &cursor, front->token_id,
                                           front->start_byte,
                                           front->end_byte)) {
      // Output full. Remaining tokens preserved for next finalize call.
      *out_token_count =
          iree_tokenizer_bpe_output_cursor_count(&cursor, output.token_ids);
      return iree_ok_status();
    }
    iree_tokenizer_bpe_window_pop_front(bpe_state, model);
  }

  *out_token_count =
      iree_tokenizer_bpe_output_cursor_count(&cursor, output.token_ids);
  bpe_state->phase = IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;
  return iree_ok_status();
}

bool iree_tokenizer_bpe_state_has_pending(
    const iree_tokenizer_model_state_t* state) {
  const iree_tokenizer_bpe_state_t* bpe_state =
      (const iree_tokenizer_bpe_state_t*)state;
  // Pending data exists when:
  //  - BYTE_LOOP: partial segment processing left unflushed window tokens.
  //  - FLUSH: output filled during final segment flush (window still has tokens
  //    to emit on subsequent finalize calls).
  return (bpe_state->phase == IREE_TOKENIZER_BPE_PHASE_BYTE_LOOP ||
          bpe_state->phase == IREE_TOKENIZER_BPE_PHASE_FLUSH) &&
         bpe_state->window.count > 0;
}

// Reclaims committed bytes from an active partial segment.
//
// When processing partial segments (last_is_partial=true), the BPE stays in
// BYTE_LOOP and accumulates tokens in its window. Frozen tokens (those far
// enough behind the processing frontier) have already been emitted. This method
// computes how many leading segment bytes are no longer needed, adjusts all
// internal byte tracking, and returns the count for ring buffer advancement.
//
// Returns 0 if no bytes can be reclaimed (not processing a partial segment,
// no frozen tokens emitted yet, or window front hasn't advanced past byte 0).
iree_host_size_t iree_tokenizer_bpe_state_reclaim(
    iree_tokenizer_model_state_t* state) {
  iree_tokenizer_bpe_state_t* bpe_state = (iree_tokenizer_bpe_state_t*)state;

  // Not processing a partial segment — nothing to reclaim.
  if (bpe_state->phase != IREE_TOKENIZER_BPE_PHASE_BYTE_LOOP) {
    return 0;
  }

  // If window is empty, all processed bytes are committed.
  if (bpe_state->window.count == 0) {
    iree_host_size_t committed = bpe_state->segment.byte_position;
    bpe_state->segment.byte_position = 0;
    return committed;
  }

  // Window has tokens — need model for circular buffer indexing.
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)state->model;
  iree_tokenizer_bpe_window_token_t* window =
      iree_tokenizer_bpe_state_window(bpe_state, model);

  // Committed = bytes before the first unflushed window token.
  iree_host_size_t committed = window[bpe_state->window.start].start_byte;
  if (committed == 0) {
    return 0;
  }

  // Adjust byte_position_in_segment.
  bpe_state->segment.byte_position -= committed;

  // Adjust all window token byte positions.
  for (iree_host_size_t i = 0; i < bpe_state->window.count; ++i) {
    iree_host_size_t index =
        (bpe_state->window.start + i) & model->window_capacity_mask;
    window[index].start_byte -= (uint32_t)committed;
    window[index].end_byte -= (uint32_t)committed;
  }

  // Adjust heap entry byte positions.
  // After BYTE_LOOP processes all segment bytes, apply_pending_merges has
  // drained most entries. Any remaining entries reference current window tokens
  // (left_start_byte >= committed). Stale entries that would underflow are
  // discarded by lazy invalidation when popped.
  for (iree_host_size_t i = 0; i < bpe_state->heap.size; ++i) {
    bpe_state->heap.entries[i].left_start_byte -= (uint32_t)committed;
  }

  return committed;
}
