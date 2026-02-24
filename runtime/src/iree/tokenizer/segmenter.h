// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Segmenter interface for pre-tokenization.
//
// Segmenters split normalized text into segments (words, subwords, punctuation)
// that the tokenization algorithm processes independently. Each segmenter type
// implements the vtable interface for pull-based streaming processing.
//
// Segments are byte ranges into the transform buffer, enabling offset tracking
// from segments back to original input positions.
//
// Design principles:
// - Pull-based: caller controls flow via output buffer capacity
// - Batched: output segment array capacity determines batch size
// - Zero allocation: state lives in caller-provided storage
// - Composable: Sequence segmenter chains multiple segmenters

#ifndef IREE_TOKENIZER_SEGMENTER_H_
#define IREE_TOKENIZER_SEGMENTER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_tokenizer_segmenter_t iree_tokenizer_segmenter_t;
typedef struct iree_tokenizer_segmenter_state_t
    iree_tokenizer_segmenter_state_t;
typedef struct iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_vtable_t;

//===----------------------------------------------------------------------===//
// Segment Types
//===----------------------------------------------------------------------===//

// A segment: byte range in the transform buffer.
// The transform buffer holds normalized text; segments index into it.
// This enables offset tracking from tokens back to original input.
typedef struct iree_tokenizer_segment_t {
  // Start byte in transform buffer.
  iree_host_size_t start;
  // End byte (exclusive) in transform buffer.
  iree_host_size_t end;
} iree_tokenizer_segment_t;

// Output buffer for segment batches.
// Groups capacity and segment array together (follows token_output_t pattern).
typedef struct iree_tokenizer_segment_output_t {
  iree_host_size_t capacity;         // Max segments to write.
  iree_tokenizer_segment_t* values;  // Output segment array.
} iree_tokenizer_segment_output_t;

// Creates a segment output buffer.
static inline iree_tokenizer_segment_output_t
iree_tokenizer_make_segment_output(iree_tokenizer_segment_t* values,
                                   iree_host_size_t capacity) {
  iree_tokenizer_segment_output_t output = {capacity, values};
  return output;
}

// Returns an empty segment output.
static inline iree_tokenizer_segment_output_t
iree_tokenizer_segment_output_empty(void) {
  iree_tokenizer_segment_output_t output = {0, NULL};
  return output;
}

//===----------------------------------------------------------------------===//
// Segmenter Base Type
//===----------------------------------------------------------------------===//

// Base segmenter structure. All concrete segmenter types embed this at
// offset 0. The vtable provides type-specific operations; state_size is
// cached at creation time for efficient state allocation.
struct iree_tokenizer_segmenter_t {
  const iree_tokenizer_segmenter_vtable_t* vtable;  // Must be at offset 0.
  // Size of state struct, cached at creation.
  iree_host_size_t state_size;
};

// Base streaming state structure. All concrete state types embed this at
// offset 0. The segmenter back-pointer enables vtable dispatch from state.
struct iree_tokenizer_segmenter_state_t {
  const iree_tokenizer_segmenter_t* segmenter;
};

//===----------------------------------------------------------------------===//
// Segmenter VTable
//===----------------------------------------------------------------------===//

// VTable for segmenter implementations.
//
// Pull-based processing model:
// - state_process() consumes normalized text, writes segment byte ranges
// - Caller controls batch size via output segment array capacity
// - Incomplete segments (waiting for boundary) buffered in state
// - state_finalize() flushes final segment
struct iree_tokenizer_segmenter_vtable_t {
  void (*destroy)(iree_tokenizer_segmenter_t* segmenter);
  iree_status_t (*state_initialize)(
      const iree_tokenizer_segmenter_t* segmenter, void* storage,
      iree_tokenizer_segmenter_state_t** out_state);
  void (*state_deinitialize)(iree_tokenizer_segmenter_state_t* state);
  iree_status_t (*state_process)(iree_tokenizer_segmenter_state_t* state,
                                 iree_string_view_t input,
                                 iree_tokenizer_segment_output_t output,
                                 iree_host_size_t* out_consumed,
                                 iree_host_size_t* out_segment_count);
  iree_status_t (*state_finalize)(iree_tokenizer_segmenter_state_t* state,
                                  iree_string_view_t remaining_input,
                                  iree_tokenizer_segment_output_t output,
                                  iree_host_size_t* out_segment_count);
  bool (*state_has_pending)(const iree_tokenizer_segmenter_state_t* state);
  iree_status_t (*state_flush)(iree_tokenizer_segmenter_state_t* state,
                               iree_tokenizer_segment_output_t output,
                               iree_host_size_t* out_segment_count,
                               iree_host_size_t* out_bytes_committed);
};

//===----------------------------------------------------------------------===//
// Segmenter Public API
//===----------------------------------------------------------------------===//

// Initializes base segmenter fields. Called by concrete implementations
// after allocating the segmenter struct.
void iree_tokenizer_segmenter_initialize(
    iree_tokenizer_segmenter_t* segmenter,
    const iree_tokenizer_segmenter_vtable_t* vtable,
    iree_host_size_t state_size);

// Frees a segmenter. Safe to call with NULL.
void iree_tokenizer_segmenter_free(iree_tokenizer_segmenter_t* segmenter);

// Returns the size of state storage required for this segmenter.
static inline iree_host_size_t iree_tokenizer_segmenter_state_size(
    const iree_tokenizer_segmenter_t* segmenter) {
  return segmenter->state_size;
}

// Initializes streaming state in caller-provided storage.
// |storage| must be at least iree_tokenizer_segmenter_state_size() bytes.
static inline iree_status_t iree_tokenizer_segmenter_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  return segmenter->vtable->state_initialize(segmenter, storage, out_state);
}

// Deinitializes streaming state (does not free storage).
static inline void iree_tokenizer_segmenter_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  if (state && state->segmenter) {
    state->segmenter->vtable->state_deinitialize(state);
  }
}

// Processes normalized text, writing segment byte ranges.
//
// Reads up to |input.size| bytes and writes up to |output.capacity| segments.
// Returns bytes consumed in |out_consumed| and segments written in
// |out_segment_count|. Segment start/end offsets are relative to the start
// of |input|; the caller is responsible for adjusting to absolute positions
// in the transform buffer.
//
// May consume less than the full input if the output segment array is full
// or the input ends mid-segment (buffered for next call).
static inline iree_status_t iree_tokenizer_segmenter_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  return state->segmenter->vtable->state_process(
      state, input, output, out_consumed, out_segment_count);
}

// Finalizes the stream, flushing any pending segment. |remaining_input|
// contains the unconsumed bytes from the last process() call. Typically
// produces 0 or 1 segment (the final incomplete segment). Segment offsets
// are relative to the start of |remaining_input|, as with process().
static inline iree_status_t iree_tokenizer_segmenter_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t remaining_input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  return state->segmenter->vtable->state_finalize(state, remaining_input,
                                                  output, out_segment_count);
}

// Returns true if state has a pending segment that would be emitted on
// finalize. Useful for checking if finalize() will produce output.
static inline bool iree_tokenizer_segmenter_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  return state->segmenter->vtable->state_has_pending(state);
}

// Forces emission of any pending segment for deadlock prevention.
//
// Called by the tokenizer when the transform buffer is full and no progress
// can be made (e.g., patterns like \p{L}+ that match unbounded runs of
// Unicode letters). Unlike finalize(), state continues to be usable after
// flush. The caller uses |out_bytes_committed| to determine how many bytes
// were actually emitted as segments; any bytes beyond this were scanned but
// not committed and should be re-provided on the next process() call.
// Returns immediately if the segmenter doesn't implement flush (vtable NULL).
static inline iree_status_t iree_tokenizer_segmenter_state_flush(
    iree_tokenizer_segmenter_state_t* state,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_segment_count,
    iree_host_size_t* out_bytes_committed) {
  if (!state->segmenter->vtable->state_flush) {
    *out_segment_count = 0;
    *out_bytes_committed = 0;
    return iree_ok_status();
  }
  return state->segmenter->vtable->state_flush(state, output, out_segment_count,
                                               out_bytes_committed);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_H_
