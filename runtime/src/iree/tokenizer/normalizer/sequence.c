// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/sequence.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Sequence Normalizer Implementation
//===----------------------------------------------------------------------===//

// Minimum batch size to feed children. Lazy normalizers (e.g., Strip with
// strip_right) need lookahead to make decisions; buffering normalizers (e.g.,
// Precompiled) accumulate patterns before emitting. 64 bytes provides enough
// context for both cases while keeping state small.
#define IREE_TOKENIZER_NORMALIZER_SEQUENCE_MIN_BATCH 64

// Scratch buffer multiplier for intermediate processing. Each tile of MIN_BATCH
// bytes is processed through all children using stack-allocated scratch. The
// scratch is split into two halves for ping-pong between stages. With 8x
// multiplier (512 bytes total, 256 per half), we can handle 4x expansion per
// stage which covers all known normalizer behaviors (NFD is ~3x worst case).
#define IREE_TOKENIZER_NORMALIZER_SEQUENCE_SCRATCH_MULTIPLIER 8

// Spillover buffer size for when processing produces more output than the
// caller can accept. During finalize_pipe, we may need to save both:
//   - Unconsumed data from current scratch half (up to half-1 bytes)
//   - Leftover input that didn't fit in scratch (up to half bytes)
// Total worst case is ~2*half - 1, so we size spillover to match scratch_size.
// With MIN_BATCH=64 and SCRATCH_MULTIPLIER=8, this is 512 bytes.
#define IREE_TOKENIZER_NORMALIZER_SEQUENCE_SPILLOVER_SIZE \
  (IREE_TOKENIZER_NORMALIZER_SEQUENCE_MIN_BATCH *         \
   IREE_TOKENIZER_NORMALIZER_SEQUENCE_SCRATCH_MULTIPLIER)

typedef struct iree_tokenizer_normalizer_sequence_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
  iree_host_size_t child_count;
  // Children stored inline (MAX_DEPTH is small and fixed).
  iree_tokenizer_normalizer_t*
      children[IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH];
  // Pre-computed child state offsets for O(1) access during state_initialize.
  iree_host_size_t
      child_state_offsets[IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH];
  // Total state size (base + all child states).
  iree_host_size_t total_state_size;
} iree_tokenizer_normalizer_sequence_t;

typedef struct iree_tokenizer_normalizer_sequence_state_t {
  iree_tokenizer_normalizer_state_t base;

  // Number of children (cached from normalizer for convenience).
  iree_host_size_t child_count;

  // Spillover buffer for when processing produces more output than the caller
  // can accept. On the next call, we drain this first before processing new
  // input. This decouples internal batch size from external output capacity.
  //
  // During finalize, spillover may hold intermediate pipe data that needs
  // further processing through remaining children (indicated by
  // finalize_pipe_stage > 0).
  uint8_t spillover[IREE_TOKENIZER_NORMALIZER_SEQUENCE_SPILLOVER_SIZE];
  iree_host_size_t spillover_length;  // Bytes stored in spillover.
  // Bytes already drained from spillover.
  iree_host_size_t spillover_offset;

  // Deferred input consumption: bytes of original input that produced the
  // spillover data. We commit this to the caller only after spillover is fully
  // drained, ensuring correct backpressure semantics.
  iree_host_size_t deferred_consumed;

  // Finalization state for waterfall cascade.
  iree_host_size_t finalize_child_index;

  // Finalize piping state: when piping finalize output through children and
  // output capacity is limited, we save intermediate data to spillover and
  // track which child to resume piping from.
  // - finalize_pipe_stage == 0: spillover is final output (just drain)
  // - finalize_pipe_stage > 0: spillover needs piping from child[stage] onward
  iree_host_size_t finalize_pipe_stage;
  // Flags to use when resuming spillover piping (SEGMENT_END or NONE).
  iree_tokenizer_normalizer_flags_t finalize_pipe_flags;

  // Child state pointers (computed during initialize from offsets).
  iree_tokenizer_normalizer_state_t*
      child_states[IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH];

  // Child states follow contiguously after this struct (properly aligned).
} iree_tokenizer_normalizer_sequence_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_sequence_vtable;

//===----------------------------------------------------------------------===//
// Allocation and Destruction
//===----------------------------------------------------------------------===//

// Returns true if the normalizer is a sequence (for flattening detection).
static bool iree_tokenizer_normalizer_is_sequence(
    const iree_tokenizer_normalizer_t* normalizer) {
  return normalizer->vtable == &iree_tokenizer_normalizer_sequence_vtable;
}

// Returns the child count of a sequence normalizer.
// Caller must ensure normalizer is a sequence.
static iree_host_size_t iree_tokenizer_normalizer_sequence_child_count(
    const iree_tokenizer_normalizer_t* normalizer) {
  const iree_tokenizer_normalizer_sequence_t* seq =
      (const iree_tokenizer_normalizer_sequence_t*)normalizer;
  return seq->child_count;
}

// Returns a child of a sequence normalizer.
// Caller must ensure normalizer is a sequence and index is valid.
static iree_tokenizer_normalizer_t* iree_tokenizer_normalizer_sequence_child(
    iree_tokenizer_normalizer_t* normalizer, iree_host_size_t index) {
  iree_tokenizer_normalizer_sequence_t* seq =
      (iree_tokenizer_normalizer_sequence_t*)normalizer;
  return seq->children[index];
}

// Clears a child slot in a sequence normalizer (for ownership transfer).
// Caller must ensure normalizer is a sequence and index is valid.
static void iree_tokenizer_normalizer_sequence_clear_child(
    iree_tokenizer_normalizer_t* normalizer, iree_host_size_t index) {
  iree_tokenizer_normalizer_sequence_t* seq =
      (iree_tokenizer_normalizer_sequence_t*)normalizer;
  seq->children[index] = NULL;
}

iree_status_t iree_tokenizer_normalizer_sequence_allocate(
    iree_tokenizer_normalizer_t* const* children, iree_host_size_t child_count,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  // Validate child count: require at least 2 children.
  // Sequences with 0 or 1 children should not be created - the JSON parser
  // handles these cases by returning NULL (empty) or the single child directly.
  if (child_count < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sequence normalizer requires at least 2 children, got %" PRIhsz
        "; use NULL for empty or pass single child directly",
        child_count);
  }
  if (child_count > IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sequence normalizer child count %" PRIhsz " exceeds maximum %" PRIu32,
        child_count, IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH);
  }

  // Validate no NULL children.
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    if (!children[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "sequence normalizer child %" PRIhsz " is NULL",
                              i);
    }
  }

  // Flatten nested sequences: if any child is itself a sequence, splice its
  // children directly into this sequence. This prevents lazy normalizers from
  // receiving insufficient lookahead context when nested sequences tile input.
  //
  // Example: Seq[Seq[NFC], Strip, Replace] becomes Seq[NFC, Strip, Replace]
  //
  // Count total flattened children first.
  iree_host_size_t flattened_count = 0;
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    if (iree_tokenizer_normalizer_is_sequence(children[i])) {
      flattened_count +=
          iree_tokenizer_normalizer_sequence_child_count(children[i]);
    } else {
      flattened_count += 1;
    }
  }

  // Validate flattened count against maximum depth.
  if (flattened_count > IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flattened sequence normalizer child count "
                            "%" PRIhsz " exceeds maximum %" PRIu32,
                            flattened_count,
                            IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH);
  }

  // Build flattened children array, transferring ownership as we go.
  // For sequence children: steal their grandchildren and free the shell.
  // For non-sequence children: transfer ownership directly.
  iree_tokenizer_normalizer_t*
      flattened_children[IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH];
  iree_host_size_t flat_index = 0;
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    if (iree_tokenizer_normalizer_is_sequence(children[i])) {
      // Splice grandchildren from nested sequence.
      iree_tokenizer_normalizer_t* nested_seq = children[i];
      iree_host_size_t nested_count =
          iree_tokenizer_normalizer_sequence_child_count(nested_seq);
      for (iree_host_size_t j = 0; j < nested_count; ++j) {
        flattened_children[flat_index++] =
            iree_tokenizer_normalizer_sequence_child(nested_seq, j);
        // Clear child slot to prevent double-free when we destroy the shell.
        iree_tokenizer_normalizer_sequence_clear_child(nested_seq, j);
      }
      // Free the now-empty sequence shell.
      iree_tokenizer_normalizer_free(nested_seq);
    } else {
      flattened_children[flat_index++] = children[i];
    }
  }
  IREE_ASSERT(flat_index == flattened_count);

  // Handle degenerate case: after flattening, we may have only 1 child.
  // Return that child directly instead of wrapping in a sequence.
  if (flattened_count == 1) {
    *out_normalizer = flattened_children[0];
    return iree_ok_status();
  }

  // Calculate total state size with proper alignment for each child.
  iree_host_size_t state_size =
      sizeof(iree_tokenizer_normalizer_sequence_state_t);
  iree_host_size_t
      child_state_offsets[IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH];
  for (iree_host_size_t i = 0; i < flattened_count; ++i) {
    state_size = iree_host_align(state_size, iree_max_align_t);
    child_state_offsets[i] = state_size;
    state_size += iree_tokenizer_normalizer_state_size(flattened_children[i]);
  }

  // Allocate the sequence normalizer.
  iree_tokenizer_normalizer_sequence_t* normalizer = NULL;
  iree_status_t status = iree_allocator_malloc(allocator, sizeof(*normalizer),
                                               (void**)&normalizer);
  if (!iree_status_is_ok(status)) {
    // Cleanup: we've taken ownership of the children, so free them on failure.
    for (iree_host_size_t i = 0; i < flattened_count; ++i) {
      iree_tokenizer_normalizer_free(flattened_children[i]);
    }
    return status;
  }

  // Initialize base.
  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_sequence_vtable,
      state_size);
  normalizer->allocator = allocator;
  normalizer->child_count = flattened_count;
  normalizer->total_state_size = state_size;

  // Store children and offsets.
  for (iree_host_size_t i = 0; i < flattened_count; ++i) {
    normalizer->children[i] = flattened_children[i];
    normalizer->child_state_offsets[i] = child_state_offsets[i];
  }

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_sequence_destroy(
    iree_tokenizer_normalizer_t* base_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_sequence_t* normalizer =
      (iree_tokenizer_normalizer_sequence_t*)base_normalizer;
  iree_allocator_t allocator = normalizer->allocator;

  // Free children in reverse order.
  for (iree_host_size_t i = normalizer->child_count; i > 0; --i) {
    iree_tokenizer_normalizer_free(normalizer->children[i - 1]);
  }

  iree_allocator_free(allocator, normalizer);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// State Management
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_sequence_state_initialize(
    const iree_tokenizer_normalizer_t* base_normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_normalizer_sequence_t* normalizer =
      (const iree_tokenizer_normalizer_sequence_t*)base_normalizer;
  iree_tokenizer_normalizer_sequence_state_t* state =
      (iree_tokenizer_normalizer_sequence_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.normalizer = base_normalizer;
  state->child_count = normalizer->child_count;

  // Initialize child states at their pre-computed offsets.
  uint8_t* storage_bytes = (uint8_t*)storage;
  for (iree_host_size_t i = 0; i < normalizer->child_count; ++i) {
    void* child_storage = storage_bytes + normalizer->child_state_offsets[i];
    iree_status_t status = iree_tokenizer_normalizer_state_initialize(
        normalizer->children[i], child_storage, &state->child_states[i]);
    if (!iree_status_is_ok(status)) {
      // Clean up already initialized children in reverse order.
      for (iree_host_size_t j = i; j > 0; --j) {
        iree_tokenizer_normalizer_state_deinitialize(
            state->child_states[j - 1]);
      }
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_sequence_state_deinitialize(
    iree_tokenizer_normalizer_state_t* base_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_sequence_state_t* state =
      (iree_tokenizer_normalizer_sequence_state_t*)base_state;
  // Deinitialize children in reverse order.
  for (iree_host_size_t i = state->child_count; i > 0; --i) {
    iree_tokenizer_normalizer_state_deinitialize(state->child_states[i - 1]);
  }
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Processing - Tiled Vertical with Spillover
//===----------------------------------------------------------------------===//

// Drains all input through a child normalizer until complete or stalled.
// Some normalizers (e.g., Precompiled with lookahead) may not consume all input
// in a single call, so this loops until all input is consumed.
//
// Parameters:
//   child_state: The child normalizer state to process through.
//   input_data: Pointer to input buffer.
//   input_length: Length of input in bytes.
//   output_data: Pointer to output buffer.
//   output_capacity: Capacity of output buffer in bytes.
//   flags: Flags to pass to child process.
//   out_written: Receives total bytes written to output.
//
// Returns OK on success. May leave unconsumed input if child stalls.
static iree_status_t iree_tokenizer_normalizer_sequence_drain_child(
    iree_tokenizer_normalizer_state_t* child_state, const char* input_data,
    iree_host_size_t input_length, char* output_data,
    iree_host_size_t output_capacity, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_host_size_t total_written = 0;
  iree_host_size_t remaining = input_length;
  const char* read_ptr = input_data;

  while (remaining > 0) {
    iree_mutable_string_view_t output = iree_make_mutable_string_view(
        output_data + total_written, output_capacity - total_written);
    iree_string_view_t input = iree_make_string_view(read_ptr, remaining);

    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
        child_state, input, output, flags, &consumed, &written));

    total_written += written;
    read_ptr += consumed;
    remaining -= consumed;

    // If child made no progress, it needs more input than available.
    if (consumed == 0 && written == 0) {
      break;
    }
  }

  *out_written = total_written;
  return iree_ok_status();
}

// Processes a single tile of data through all children in the sequence.
// Uses the provided scratch buffer for intermediate stages, writing final
// output to the output buffer. If the result exceeds output capacity, the
// excess is stored in the spillover buffer.
//
// Parameters:
//   state: Sequence state (for child_states and spillover)
//   input: Input data for child[0]
//   scratch: Scratch buffer for intermediate stages (must be >= input.size * 3
//            to handle worst-case expansion like NFD decomposition)
//   scratch_size: Size of scratch buffer
//   output: Final output buffer
//   flags: Flags to propagate to children
//   out_consumed: Bytes consumed from input
//   out_written: Bytes written to output (excludes spillover)
//
// Returns OK on success. If spillover is used, caller must drain it before
// processing more input.
static iree_status_t iree_tokenizer_normalizer_sequence_process_tile(
    iree_tokenizer_normalizer_sequence_state_t* state, iree_string_view_t input,
    uint8_t* scratch, iree_host_size_t scratch_size,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  *out_consumed = 0;
  *out_written = 0;

  if (input.size == 0) {
    return iree_ok_status();
  }

  // Process through child[0].
  // Use first half of scratch for child[0]'s output.
  iree_host_size_t half = scratch_size / 2;
  uint8_t* buffer_a = scratch;
  uint8_t* buffer_b = scratch + half;

  iree_mutable_string_view_t child0_output =
      iree_make_mutable_string_view((char*)buffer_a, half);
  iree_host_size_t child0_consumed = 0;
  iree_host_size_t child0_written = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
      state->child_states[0], input, child0_output, flags, &child0_consumed,
      &child0_written));

  if (child0_consumed == 0 && child0_written == 0) {
    // No progress - child[0] needs more input (lazy normalizer).
    return iree_ok_status();
  }

  *out_consumed = child0_consumed;

  if (child0_written == 0) {
    // Child[0] consumed input but produced no output (e.g., stripping).
    return iree_ok_status();
  }

  // Process through middle children using ping-pong within scratch.
  uint8_t* read_buffer = buffer_a;
  iree_host_size_t read_count = child0_written;

  for (iree_host_size_t stage = 1; stage < state->child_count - 1; ++stage) {
    uint8_t* write_buffer = (read_buffer == buffer_a) ? buffer_b : buffer_a;

    iree_host_size_t stage_written = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_sequence_drain_child(
        state->child_states[stage], (const char*)read_buffer, read_count,
        (char*)write_buffer, half, flags, &stage_written));

    read_buffer = write_buffer;
    read_count = stage_written;

    if (read_count == 0) {
      // Middle stage consumed all but produced nothing.
      return iree_ok_status();
    }
  }

  // Process through final child, writing to output + spillover if needed.
  // If output is large enough for worst-case expansion, write directly.
  if (output.size >= read_count * 3) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_sequence_drain_child(
        state->child_states[state->child_count - 1], (const char*)read_buffer,
        read_count, output.data, output.size, flags, out_written));
    return iree_ok_status();
  }

  // Output might be too small - write to scratch first, then split.
  // Use the OTHER buffer (not read_buffer) to avoid overwriting input.
  uint8_t* final_scratch = (read_buffer == buffer_a) ? buffer_b : buffer_a;

  iree_host_size_t total_written = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_sequence_drain_child(
      state->child_states[state->child_count - 1], (const char*)read_buffer,
      read_count, (char*)final_scratch, half, flags, &total_written));

  // Copy what fits to output, rest to spillover.
  iree_host_size_t to_output =
      total_written < output.size ? total_written : output.size;
  memcpy(output.data, final_scratch, to_output);
  *out_written = to_output;

  if (total_written > to_output) {
    // Store excess in spillover.
    iree_host_size_t excess = total_written - to_output;
    IREE_ASSERT(excess <= IREE_TOKENIZER_NORMALIZER_SEQUENCE_SPILLOVER_SIZE,
                "spillover buffer overflow");
    memcpy(state->spillover, final_scratch + to_output, excess);
    state->spillover_length = excess;
    state->spillover_offset = 0;
  }

  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_sequence_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_sequence_state_t* state =
      (iree_tokenizer_normalizer_sequence_state_t*)base_state;

  *out_consumed = 0;
  *out_written = 0;

  IREE_ASSERT(state->child_count >= 2);

  iree_host_size_t total_written = 0;
  iree_mutable_string_view_t remaining_output = output;

  // Phase 1: Drain spillover from a previous call.
  // When the previous call produced more output than the caller could accept,
  // the excess was stored in spillover. Drain it first before processing new
  // input.
  if (state->spillover_length > 0) {
    iree_host_size_t spillover_remaining =
        state->spillover_length - state->spillover_offset;
    iree_host_size_t to_copy = spillover_remaining < remaining_output.size
                                   ? spillover_remaining
                                   : remaining_output.size;
    memcpy(remaining_output.data, state->spillover + state->spillover_offset,
           to_copy);
    state->spillover_offset += to_copy;
    total_written += to_copy;
    remaining_output.data += to_copy;
    remaining_output.size -= to_copy;

    if (state->spillover_offset < state->spillover_length) {
      // Still have spillover data - don't consume any new input.
      *out_consumed = 0;
      *out_written = total_written;
      return iree_ok_status();
    }

    // Spillover fully drained. Commit the deferred input consumption.
    *out_consumed = state->deferred_consumed;
    state->deferred_consumed = 0;
    state->spillover_length = 0;
    state->spillover_offset = 0;
  }

  // Phase 2: Process new input through the child chain.
  // Key insight: enforce minimum batch size to give lazy normalizers enough
  // lookahead context. Don't let a small output buffer shrink the input window.
  iree_string_view_t remaining_input = input;
  remaining_input.data += *out_consumed;
  remaining_input.size -= *out_consumed;

  // Allocate scratch buffer on stack for intermediate processing.
  // Size: MIN_BATCH * SCRATCH_MULTIPLIER to handle worst-case expansion.
  uint8_t scratch[IREE_TOKENIZER_NORMALIZER_SEQUENCE_MIN_BATCH *
                  IREE_TOKENIZER_NORMALIZER_SEQUENCE_SCRATCH_MULTIPLIER];
  iree_host_size_t scratch_size = sizeof(scratch);

  while (remaining_input.size > 0 && remaining_output.size > 0) {
    // Calculate tile size: enforce minimum batch for lazy normalizer context.
    // The minimum ensures normalizers get enough lookahead even when the
    // caller provides a tiny output buffer.
    iree_host_size_t tile_size = remaining_input.size;
    if (tile_size > IREE_TOKENIZER_NORMALIZER_SEQUENCE_MIN_BATCH) {
      // Limit tile to what we can handle in scratch.
      tile_size = IREE_TOKENIZER_NORMALIZER_SEQUENCE_MIN_BATCH;
    }

    // Process one tile through all children.
    iree_string_view_t tile_input =
        iree_make_string_view(remaining_input.data, tile_size);
    iree_host_size_t tile_consumed = 0;
    iree_host_size_t tile_written = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_sequence_process_tile(
        state, tile_input, scratch, scratch_size, remaining_output, flags,
        &tile_consumed, &tile_written));

    if (tile_consumed == 0 && tile_written == 0) {
      // No progress - child[0] needs more input (shouldn't happen with
      // MIN_BATCH, but handle gracefully).
      break;
    }

    remaining_input.data += tile_consumed;
    remaining_input.size -= tile_consumed;
    total_written += tile_written;
    remaining_output.data += tile_written;
    remaining_output.size -= tile_written;

    // Check if spillover was used - if so, defer the consumption and return.
    if (state->spillover_length > 0) {
      state->deferred_consumed = tile_consumed;
      *out_written = total_written;
      return iree_ok_status();
    }

    // Tile completed without spillover - commit consumption.
    *out_consumed += tile_consumed;
  }

  *out_written = total_written;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

// Helper to pipe data through children[start_stage..N-1] during finalize.
// Uses spillover for data that can't fit in output.
// |flags| are passed to children during process - caller determines whether
// SEGMENT_END should be set based on whether more data will follow.
static iree_status_t iree_tokenizer_normalizer_sequence_finalize_pipe(
    iree_tokenizer_normalizer_sequence_state_t* state,
    const uint8_t* input_data, iree_host_size_t input_length,
    iree_host_size_t start_stage, iree_tokenizer_normalizer_flags_t flags,
    iree_mutable_string_view_t* remaining_output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  // Scratch for ping-pong between stages.
  uint8_t scratch[IREE_TOKENIZER_NORMALIZER_SEQUENCE_MIN_BATCH *
                  IREE_TOKENIZER_NORMALIZER_SEQUENCE_SCRATCH_MULTIPLIER];
  iree_host_size_t scratch_size = sizeof(scratch);
  iree_host_size_t half = scratch_size / 2;
  uint8_t* buffer_a = scratch;
  uint8_t* buffer_b = scratch + half;

  // Copy input to buffer_a to start ping-pong.
  iree_host_size_t to_process = input_length < half ? input_length : half;
  memcpy(buffer_a, input_data, to_process);
  uint8_t* read_buf = buffer_a;
  iree_host_size_t read_count = to_process;

  // Track if we have leftover input that didn't fit in scratch.
  const uint8_t* leftover = input_data + to_process;
  iree_host_size_t leftover_count = input_length - to_process;

  // Process through children [start_stage..N-1] with ping-pong.
  // Some normalizers (e.g., Precompiled) may not consume all input in a single
  // call, so we loop until each non-final stage fully processes its input.
  for (iree_host_size_t j = start_stage; j < state->child_count; ++j) {
    bool j_is_last = (j == state->child_count - 1);
    iree_mutable_string_view_t stage_output;
    uint8_t* write_buf;

    if (j_is_last) {
      stage_output = *remaining_output;
      write_buf = NULL;

      iree_string_view_t stage_input =
          iree_make_string_view((const char*)read_buf, read_count);
      iree_host_size_t stage_consumed = 0;
      iree_host_size_t stage_written = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
          state->child_states[j], stage_input, stage_output, flags,
          &stage_consumed, &stage_written));

      *out_written += stage_written;
      remaining_output->data += stage_written;
      remaining_output->size -= stage_written;

      // Check if we have unconsumed data or leftover to save for next call.
      iree_host_size_t unconsumed =
          (stage_consumed < read_count) ? (read_count - stage_consumed) : 0;
      iree_host_size_t total_remaining = unconsumed + leftover_count;

      if (total_remaining > 0) {
        // Save remaining data to spillover for next call.
        IREE_ASSERT(total_remaining <=
                    IREE_TOKENIZER_NORMALIZER_SEQUENCE_SPILLOVER_SIZE);
        if (unconsumed > 0) {
          memcpy(state->spillover, read_buf + stage_consumed, unconsumed);
        }
        if (leftover_count > 0) {
          // Use memmove because leftover may point into spillover
          // (overlapping).
          memmove(state->spillover + unconsumed, leftover, leftover_count);
        }
        state->spillover_length = total_remaining;
        state->spillover_offset = 0;
        state->finalize_pipe_stage = j;      // Resume from last stage.
        state->finalize_pipe_flags = flags;  // Use same flags on resume.
      } else {
        // All data processed - clear spillover state.
        state->spillover_length = 0;
        state->spillover_offset = 0;
        state->finalize_pipe_stage = 0;
        state->finalize_pipe_flags = IREE_TOKENIZER_NORMALIZER_FLAG_NONE;
      }
    } else {
      // Non-final stage: drain all input through this child.
      write_buf = (read_buf == buffer_a) ? buffer_b : buffer_a;

      iree_host_size_t stage_written = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_sequence_drain_child(
          state->child_states[j], (const char*)read_buf, read_count,
          (char*)write_buf, half, flags, &stage_written));

      read_buf = write_buf;
      read_count = stage_written;
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_sequence_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_sequence_state_t* state =
      (iree_tokenizer_normalizer_sequence_state_t*)base_state;

  *out_written = 0;

  // Reject finalize while process() has deferred consumption pending.
  if (state->deferred_consumed > 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "sequence normalizer has %" PRIhsz
        " bytes of deferred input consumption; "
        "drain process() with more output capacity before finalizing",
        state->deferred_consumed);
  }

  IREE_ASSERT(state->child_count >= 2);

  iree_host_size_t total_written = 0;
  iree_mutable_string_view_t remaining_output = output;

  // First, handle spillover from a previous finalize call.
  if (state->spillover_length > 0) {
    iree_host_size_t spillover_remaining =
        state->spillover_length - state->spillover_offset;

    if (state->finalize_pipe_stage > 0) {
      // Spillover contains intermediate data - pipe through remaining children.
      // Use the flags that were saved when this spillover was created.
      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_sequence_finalize_pipe(
          state, state->spillover + state->spillover_offset,
          spillover_remaining, state->finalize_pipe_stage,
          state->finalize_pipe_flags, &remaining_output, &total_written));

      if (state->spillover_length > 0) {
        // Still have data to pipe.
        *out_written = total_written;
        return iree_ok_status();
      }
      state->finalize_pipe_stage = 0;
    } else {
      // Spillover contains final output - just drain to caller.
      iree_host_size_t to_copy = spillover_remaining < remaining_output.size
                                     ? spillover_remaining
                                     : remaining_output.size;
      memcpy(remaining_output.data, state->spillover + state->spillover_offset,
             to_copy);
      state->spillover_offset += to_copy;
      total_written += to_copy;
      remaining_output.data += to_copy;
      remaining_output.size -= to_copy;

      if (state->spillover_offset < state->spillover_length) {
        *out_written = total_written;
        return iree_ok_status();
      }
    }
    state->spillover_length = 0;
    state->spillover_offset = 0;
  }

  // Scratch buffer for child finalize output.
  uint8_t scratch[IREE_TOKENIZER_NORMALIZER_SEQUENCE_MIN_BATCH *
                  IREE_TOKENIZER_NORMALIZER_SEQUENCE_SCRATCH_MULTIPLIER];
  iree_host_size_t scratch_size = sizeof(scratch);

  // Waterfall cascade: finalize child[i], pipe through children[i+1..N-1].
  iree_host_size_t i = state->finalize_child_index;
  while (i < state->child_count) {
    bool is_last = (i == state->child_count - 1);

    iree_mutable_string_view_t child_output;
    if (is_last) {
      child_output = remaining_output;
    } else {
      child_output =
          iree_make_mutable_string_view((char*)scratch, scratch_size);
    }

    iree_host_size_t child_written = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_finalize(
        state->child_states[i], child_output, &child_written));

    if (is_last) {
      total_written += child_written;
      remaining_output.data += child_written;
      remaining_output.size -= child_written;

      if (iree_tokenizer_normalizer_state_has_pending(state->child_states[i])) {
        state->finalize_child_index = i;
        *out_written = total_written;
        return iree_ok_status();
      }
      ++i;
    } else if (child_written > 0) {
      // Determine if this child is done with finalization.
      bool child_done =
          !iree_tokenizer_normalizer_state_has_pending(state->child_states[i]);

      // Only pass SEGMENT_END when ALL non-final children are done.
      // This happens when:
      //   1. Current child (i) is done (no more output from it)
      //   2. i is the last non-final child (i == N-2)
      // When earlier children finalize, later non-final children (i+1..N-2)
      // may still produce finalize output, so we can't signal end yet.
      bool last_nonfinal_child = (i == state->child_count - 2);
      iree_tokenizer_normalizer_flags_t pipe_flags =
          (child_done && last_nonfinal_child)
              ? IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END
              : IREE_TOKENIZER_NORMALIZER_FLAG_NONE;

      // Pipe finalize output through children[i+1..N-1].
      iree_host_size_t pipe_written = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_sequence_finalize_pipe(
          state, scratch, child_written, i + 1, pipe_flags, &remaining_output,
          &pipe_written));
      total_written += pipe_written;

      // If spillover was used, try to drain it while output has capacity.
      // Spillover contains intermediate data - use same flags as original pipe.
      while (state->spillover_length > 0 && remaining_output.size > 0) {
        iree_host_size_t spillover_remaining =
            state->spillover_length - state->spillover_offset;
        iree_host_size_t drain_written = 0;
        IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_sequence_finalize_pipe(
            state, state->spillover + state->spillover_offset,
            spillover_remaining, state->finalize_pipe_stage, pipe_flags,
            &remaining_output, &drain_written));
        total_written += drain_written;
      }

      // If spillover still has data and output is full, return.
      if (state->spillover_length > 0) {
        state->finalize_child_index = i;
        *out_written = total_written;
        return iree_ok_status();
      }
      state->finalize_pipe_stage = 0;

      if (child_done) {
        ++i;
      }
    } else {
      if (!iree_tokenizer_normalizer_state_has_pending(
              state->child_states[i])) {
        ++i;
      }
    }
  }

  // All done - reset state for potential reuse.
  state->finalize_child_index = 0;
  state->finalize_pipe_stage = 0;
  state->finalize_pipe_flags = IREE_TOKENIZER_NORMALIZER_FLAG_NONE;

  *out_written = total_written;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Has Pending
//===----------------------------------------------------------------------===//

static bool iree_tokenizer_normalizer_sequence_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_sequence_state_t* state =
      (const iree_tokenizer_normalizer_sequence_state_t*)base_state;

  // Check if we have spillover data waiting to be drained.
  if (state->spillover_length > state->spillover_offset) {
    return true;
  }

  // Check if any child has pending data.
  for (iree_host_size_t i = 0; i < state->child_count; ++i) {
    if (iree_tokenizer_normalizer_state_has_pending(state->child_states[i])) {
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_sequence_vtable = {
        .destroy = iree_tokenizer_normalizer_sequence_destroy,
        .state_initialize = iree_tokenizer_normalizer_sequence_state_initialize,
        .state_deinitialize =
            iree_tokenizer_normalizer_sequence_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_sequence_state_process,
        .state_finalize = iree_tokenizer_normalizer_sequence_state_finalize,
        .state_has_pending =
            iree_tokenizer_normalizer_sequence_state_has_pending,
};
