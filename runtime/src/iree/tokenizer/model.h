// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Model interface for subword tokenization.
//
// Models implement the core tokenization step: converting segments
// (from the segmenter) to token IDs. Examples: BPE, WordPiece, Unigram.
//
// The encode pipeline is:
//   Input → Normalizer → Segmenter → Model → Token IDs
//
// Design principles:
// - Pull-based: output buffer capacity drives processing
// - Streaming: can process segments incrementally (greedy left-to-right)
// - Zero allocation: state lives in caller-provided storage
//
// Streaming model:
// - Model tracks position within current segment
// - Greedy longest-match emits tokens as they're determined
// - No need to see entire segment before emitting tokens

#ifndef IREE_TOKENIZER_MODEL_H_
#define IREE_TOKENIZER_MODEL_H_

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_tokenizer_model_t iree_tokenizer_model_t;
typedef struct iree_tokenizer_model_state_t iree_tokenizer_model_state_t;
typedef struct iree_tokenizer_model_vtable_t iree_tokenizer_model_vtable_t;

//===----------------------------------------------------------------------===//
// Segment List (for batched input)
//===----------------------------------------------------------------------===//

// List of segments for batched model input.
// Segments are byte ranges into the transform buffer.
typedef struct iree_tokenizer_segment_list_t {
  iree_host_size_t count;
  const iree_tokenizer_segment_t* values;
  // When true, the last segment in the list is incomplete — more bytes will be
  // appended in subsequent calls. The model must use an incremental processing
  // path (e.g., BYTE_LOOP for BPE) and must not flush/finalize the last
  // segment. It reports segment_complete=false and preserves internal state
  // (window, heap, byte position) for continuation.
  //
  // This enables streaming encode for inputs that exceed the ring buffer's
  // capacity, such as split=false models where the entire input is one segment.
  // The frozen token theorem guarantees correctness: tokens sufficiently far
  // from the processing frontier are emitted incrementally.
  bool last_is_partial;
} iree_tokenizer_segment_list_t;

// Creates a segment list from an array and count.
static inline iree_tokenizer_segment_list_t iree_tokenizer_make_segment_list(
    const iree_tokenizer_segment_t* values, iree_host_size_t count) {
  iree_tokenizer_segment_list_t list = {count, values,
                                        /*last_is_partial=*/false};
  return list;
}

// Returns an empty segment list.
static inline iree_tokenizer_segment_list_t iree_tokenizer_segment_list_empty(
    void) {
  iree_tokenizer_segment_list_t list = {0, NULL, /*last_is_partial=*/false};
  return list;
}

//===----------------------------------------------------------------------===//
// Model Base Type
//===----------------------------------------------------------------------===//

// Base model structure. All concrete model types embed this at
// offset 0. The vtable provides type-specific operations; state_size is
// cached at creation time for efficient state allocation.
struct iree_tokenizer_model_t {
  const iree_tokenizer_model_vtable_t* vtable;  // Must be at offset 0.
  // Size of state struct, cached at creation.
  iree_host_size_t state_size;
  // Human-readable model type (e.g., "BPE").
  iree_string_view_t type_name;
};

// Base streaming state structure. All concrete state types embed this at
// offset 0. The model back-pointer enables vtable dispatch from state.
//
// Model state tracks position within the current segment for streaming
// tokenization. Greedy BPE processes left-to-right, emitting tokens as they
// become determined, so state is O(1) - just position tracking.
struct iree_tokenizer_model_state_t {
  const iree_tokenizer_model_t* model;
};

//===----------------------------------------------------------------------===//
// Model VTable
//===----------------------------------------------------------------------===//

// VTable for model implementations.
//
// Streaming model:
// - state_encode() processes segments incrementally
// - Model tracks current segment index and position within segment
// - Greedy longest-match determines tokens left-to-right
// - Tokens emitted immediately as they're determined (no buffering)
// - state_finalize() handles any trailing bytes that need special handling
struct iree_tokenizer_model_vtable_t {
  void (*destroy)(iree_tokenizer_model_t* model);
  iree_status_t (*state_initialize)(const iree_tokenizer_model_t* model,
                                    void* storage,
                                    iree_tokenizer_model_state_t** out_state);
  void (*state_deinitialize)(iree_tokenizer_model_state_t* state);
  iree_status_t (*state_encode)(iree_tokenizer_model_state_t* state,
                                iree_const_byte_span_t transform_buffer,
                                iree_tokenizer_segment_list_t segments,
                                iree_tokenizer_token_output_t output,
                                iree_host_size_t* out_segments_consumed,
                                iree_host_size_t* out_token_count);
  iree_status_t (*state_finalize)(iree_tokenizer_model_state_t* state,
                                  iree_tokenizer_token_output_t output,
                                  iree_host_size_t* out_token_count);
  bool (*state_has_pending)(const iree_tokenizer_model_state_t* state);
  iree_host_size_t (*state_reclaim)(iree_tokenizer_model_state_t* state);
  iree_status_t (*get_token_string)(const iree_tokenizer_model_t* model,
                                    iree_tokenizer_token_id_t token_id,
                                    iree_string_view_t* out_string);
};

//===----------------------------------------------------------------------===//
// Model Public API
//===----------------------------------------------------------------------===//

// Initializes base model fields. Called by concrete implementations
// after allocating the model struct. |type_name| must be a static string
// (e.g., IREE_SV("BPE")) that outlives the model.
void iree_tokenizer_model_initialize(
    iree_tokenizer_model_t* model, const iree_tokenizer_model_vtable_t* vtable,
    iree_host_size_t state_size, iree_string_view_t type_name);

// Frees a model. Safe to call with NULL.
void iree_tokenizer_model_free(iree_tokenizer_model_t* model);

// Returns the size of state storage required for this model.
static inline iree_host_size_t iree_tokenizer_model_state_size(
    const iree_tokenizer_model_t* model) {
  return model->state_size;
}

// Initializes streaming state in caller-provided storage.
// |storage| must be at least iree_tokenizer_model_state_size() bytes.
static inline iree_status_t iree_tokenizer_model_state_initialize(
    const iree_tokenizer_model_t* model, void* storage,
    iree_tokenizer_model_state_t** out_state) {
  return model->vtable->state_initialize(model, storage, out_state);
}

// Deinitializes streaming state (does not free storage).
static inline void iree_tokenizer_model_state_deinitialize(
    iree_tokenizer_model_state_t* state) {
  if (state && state->model) {
    state->model->vtable->state_deinitialize(state);
  }
}

// Encodes segments to token IDs, streaming within each segment.
//
// |transform_buffer| is the normalized text backing the |segments| indices.
// All segment start/end offsets must be valid indices into this buffer.
// Processes segments using greedy longest-match, tracking current segment
// index and byte position in state. Tokens are emitted as they're determined
// without future lookahead.
//
// Returns the number of fully-processed segments in |out_segments_consumed|
// and the number of tokens written in |out_token_count|. May consume fewer
// segments than provided if the output buffer fills mid-segment. In that case,
// state remembers the position within the partial segment and the next call
// continues from there. A partially-processed segment is not counted in
// |out_segments_consumed| until fully complete.
static inline iree_status_t iree_tokenizer_model_state_encode(
    iree_tokenizer_model_state_t* state,
    iree_const_byte_span_t transform_buffer,
    iree_tokenizer_segment_list_t segments,
    iree_tokenizer_token_output_t output,
    iree_host_size_t* out_segments_consumed,
    iree_host_size_t* out_token_count) {
  return state->model->vtable->state_encode(state, transform_buffer, segments,
                                            output, out_segments_consumed,
                                            out_token_count);
}

// Finalizes encoding, handling any trailing state. Call after all segments
// have been provided via state_encode(). Most models produce no output here
// since greedy emits as it goes; some may emit a final token for incomplete
// sequences. May need multiple calls if the output buffer is too small.
static inline iree_status_t iree_tokenizer_model_state_finalize(
    iree_tokenizer_model_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* out_token_count) {
  return state->model->vtable->state_finalize(state, output, out_token_count);
}

// Returns true if state has pending data that would produce output on
// finalize. Useful for checking if more finalize() calls are needed.
static inline bool iree_tokenizer_model_state_has_pending(
    const iree_tokenizer_model_state_t* state) {
  return state->model->vtable->state_has_pending(state);
}

// Reclaims committed bytes from the current partial segment.
//
// When processing a partial segment (last_is_partial=true), the model
// accumulates frozen tokens in its window. This computes how many leading
// segment bytes are fully committed (all tokens covering those bytes have
// been emitted), adjusts all internal byte position tracking by subtracting
// the committed count, and returns the number of bytes reclaimed. The caller
// advances the ring buffer's read position by the returned count. Returns 0
// if no bytes can be reclaimed or the model is not processing a partial
// segment.
static inline iree_host_size_t iree_tokenizer_model_state_reclaim(
    iree_tokenizer_model_state_t* state) {
  return state->model->vtable->state_reclaim(state);
}

// Looks up the string representation of a token ID. The returned string is
// valid for the lifetime of the model (typically backed by vocab storage).
// Returns IREE_STATUS_NOT_FOUND if |token_id| is not in the vocabulary.
static inline iree_status_t iree_tokenizer_model_get_token_string(
    const iree_tokenizer_model_t* model, iree_tokenizer_token_id_t token_id,
    iree_string_view_t* out_string) {
  return model->vtable->get_token_string(model, token_id, out_string);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_MODEL_H_
