// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/tokenizer.h"

#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/decoder.h"
#include "iree/tokenizer/decoder/byte_fallback.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/postprocessor.h"
#include "iree/tokenizer/segmenter.h"
#include "iree/tokenizer/special_tokens.h"
#include "iree/tokenizer/vocab/vocab.h"

//===----------------------------------------------------------------------===//
// Configuration Constants
//===----------------------------------------------------------------------===//

// Default segment batch size. This determines how many segments are buffered
// between the segmenter and model stages. Larger values amortize vtable
// call overhead but use more state storage.
#define IREE_TOKENIZER_DEFAULT_SEGMENT_BATCH_SIZE 64

// Default string batch size for decode. This determines how many token strings
// are looked up from vocab before feeding to the decoder. Larger values
// amortize vtable call overhead but use more state storage (16 bytes per slot
// on 64-bit systems for iree_string_view_t).
#define IREE_TOKENIZER_DEFAULT_STRING_BATCH_SIZE 64

//===----------------------------------------------------------------------===//
// Internal Structures
//===----------------------------------------------------------------------===//

// Pre-decoded token string table. Built at tokenizer construction time when
// the decoder has STATELESS capability, eliminating runtime decoder work.
//
// Single slab allocation: [offsets_array][data_blob]
//   offsets: (vocab_capacity + 1) entries, prefix-sum style.
//   data: flat blob of decoded byte strings.
// Token id's decoded bytes are at data[offsets[id] .. offsets[id+1]).
typedef struct iree_tokenizer_pre_decoded_t {
  // Combined allocation (NULL if not pre-decodable).
  void* slab;
  uint32_t* offsets;
  const uint8_t* data;
  // For position-sensitive decoders (Metaspace, WordPiece): the table stores
  // the "rest" form (with leading space). The first token in a stream skips
  // the leading space byte if present.
  bool position_sensitive;
  // When true, the decoder chain contains ByteFallback and byte tokens
  // (<0xHH>) need an inline accumulator at decode time. Non-byte tokens
  // are pre-decoded normally; byte tokens store a single raw byte value
  // in the data table.
  bool has_byte_fallback;
  // Byte token ID range (inclusive). Byte tokens are nearly contiguous but
  // may have small gaps (e.g., Gemma has a literal tab token at ID 226 in
  // the middle of its byte token range 217-472). The bitmap below handles
  // gaps. Decode-time fast rejection:
  //   id >= byte_token_first_id && id <= byte_token_last_id
  // Then bitmap check for gap handling:
  //   byte_token_bitmap[(id - byte_token_first_id) / 8]
  //     & (1u << ((id - byte_token_first_id) % 8))
  // Only valid when has_byte_fallback is true.
  int32_t byte_token_first_id;
  int32_t byte_token_last_id;
  // Bitmap marking which IDs in [first_id, last_id] are byte tokens.
  // Bit N corresponds to ID (byte_token_first_id + N).
  // Max 256 byte tokens → 32 bytes.
  uint8_t byte_token_bitmap[32];
} iree_tokenizer_pre_decoded_t;

// Internal tokenizer structure.
// Opaque to callers; built via iree_tokenizer_builder_build().
struct iree_tokenizer_t {
  iree_allocator_t allocator;

  // Pipeline components (owned).
  iree_tokenizer_normalizer_t* normalizer;
  iree_tokenizer_segmenter_t* segmenter;
  iree_tokenizer_model_t* model;
  iree_tokenizer_decoder_t* decoder;

  // Post-processor (value type, zero-initialized = no-op).
  iree_tokenizer_postprocessor_t postprocessor;

  // Vocabulary (owned). Must be freed AFTER model since model references it.
  iree_tokenizer_vocab_t* vocab;

  // For encode: segment batching between segmenter and model.
  iree_host_size_t segment_batch_size;
  // For decode: token string batching between vocab lookup and decoder.
  iree_host_size_t string_batch_size;

  // Pre-decoded token strings (owned, zero-initialized if not available).
  iree_tokenizer_pre_decoded_t pre_decoded;

  // Special tokens for input preprocessing (value type).
  // Tokens with special: true that must be matched in raw input BEFORE
  // normalization runs. Matched via two-level prefix index for O(1) rejection.
  iree_tokenizer_special_tokens_t special_tokens;

  // Special tokens for post-normalization matching (value type).
  // Tokens with normalized: true that are matched AFTER normalization runs.
  iree_tokenizer_special_tokens_t special_tokens_post_norm;
};

// Internal encode state structure.
// Layout in state_storage:
//   [encode_state_t]
//   [normalizer_state (normalizer->state_size bytes)]
//   [segmenter_state (segmenter->state_size bytes)]
//   [model_state (model->state_size bytes)]
//   [segment_buffer (segment_batch_size * sizeof(segment_t))]
struct iree_tokenizer_encode_state_t {
  const iree_tokenizer_t* tokenizer;

  // User-provided buffers.
  iree_byte_span_t transform_buffer;
  iree_tokenizer_offset_run_list_t offset_runs;

  // Flags from initialization.
  iree_tokenizer_encode_flags_t flags;

  // Special token partial match state for streaming. Runs BEFORE the normalizer
  // to match literal tokens like <|endoftext|> in raw input. Tracks partial
  // matches that span chunk boundaries so they can be completed or recovered.
  iree_tokenizer_special_tokens_encode_state_t special_token_match;

  // Post-normalization special token match state. Runs AFTER the normalizer
  // to match tokens with normalized=true in the transformed output.
  iree_tokenizer_special_tokens_encode_state_t special_token_match_post;

  // Ring buffer position tracking. All positions are logical (cumulative).
  // Physical buffer index = logical_position & capacity_mask.
  // Invariants:
  //   read_position <= segmenter_view_start <= write_position
  //   write_position - read_position <= (capacity_mask + 1)
  //
  // Buffer capacity - 1, for fast & instead of %. Must be power of two minus 1.
  iree_host_size_t capacity_mask;
  // Oldest byte still needed. Bytes before this position can be overwritten.
  iree_host_size_t read_position;
  // Where normalizer writes next.
  iree_host_size_t write_position;
  // Start of segmenter's view. The segmenter sees [segmenter_view_start,
  // write_position). Used for view management and reclaim tracking. Advances
  // when segmenter consumes bytes or when post-norm special tokens skip bytes.
  iree_host_size_t segmenter_view_start;

  // Segment buffer (points into state_storage after stage states).
  iree_tokenizer_segment_t* segments;
  iree_host_size_t segment_count;
  iree_host_size_t segments_consumed;

  // Pipeline stage states (point into state_storage).
  iree_tokenizer_normalizer_state_t* normalizer_state;
  iree_tokenizer_segmenter_state_t* segmenter_state;
  iree_tokenizer_model_state_t* model_state;

  // Postprocessor state. When phase is IDLE, all postprocessor operations are
  // no-ops (either ADD_SPECIAL_TOKENS not set or template has no tokens).
  iree_tokenizer_postprocessor_encode_state_t postprocessor;

  // True when the ring buffer is full and the segmenter hasn't produced
  // boundaries (split=false). In this mode, the pump feeds available ring data
  // directly to BPE as a partial segment, bypassing the segmenter.
  bool has_partial_segment;

  // Deferred pre-normalization special token emission.
  //
  // When a special token is matched but the pipeline still has content (ring
  // buffer or pending segments), we defer emission until that content is
  // tokenized. This ensures correct output ordering: "hello<|endoftext|>world"
  // produces [hello_tokens..., <|endoftext|>, world_tokens...], not
  // [<|endoftext|>, hello_tokens..., world_tokens...].
  //
  // -1 means no pending token. When >= 0, emit after pipeline flushes.
  iree_tokenizer_token_id_t pending_special_token;

  // True when a special token consumed position 0 of the input. Used to signal
  // the prepend normalizer to skip prepending for prepend_scheme="first"
  // semantics: text following a position-0 special token is NOT "first".
  bool first_consumed_by_special_token;

  // True when in finalize mode. Post-norm special token matching uses this to
  // treat NEED_MORE from speculative checks as NO_MATCH (no more input coming,
  // so a partial prefix can never become a complete special token).
  bool in_finalize_mode;
};

// Internal decode state structure.
// Layout in state_storage:
//   [decode_state_t]
//   [decoder_state (decoder->state_size bytes)]
//   [string_buffer (string_batch_size * sizeof(iree_string_view_t))]
struct iree_tokenizer_decode_state_t {
  const iree_tokenizer_t* tokenizer;

  // Decode behavior flags (stored at initialization, applied during feed).
  iree_tokenizer_decode_flags_t flags;

  // True for the first token in a stream. Used by position-sensitive
  // pre-decoded fast path to strip the leading space from the first token's
  // output.
  bool is_first_token;

  // Inline byte accumulator for hybrid pre-decoded path with ByteFallback.
  // Accumulates raw bytes from byte tokens (<0xHH>) until a complete UTF-8
  // sequence is formed, then emits the validated bytes. Only used when
  // pre_decoded.has_byte_fallback is true.
  uint8_t byte_fallback_pending[4];
  uint8_t byte_fallback_pending_count;
  uint8_t byte_fallback_expected_length;

  // Pipeline stage state (points into state_storage).
  // NULL when using the pre-decoded fast path.
  iree_tokenizer_decoder_state_t* decoder_state;

  // String buffer for batching vocab lookups (points into state_storage).
  // Filled from vocab lookup, consumed by decoder.
  // NULL when using the pre-decoded fast path.
  iree_string_view_t* string_buffer;
  iree_host_size_t string_count;      // Strings currently in buffer.
  iree_host_size_t strings_consumed;  // Strings processed by decoder.
};

//===----------------------------------------------------------------------===//
// Ring Buffer Helpers
//===----------------------------------------------------------------------===//

// Converts a logical (cumulative) position to a physical buffer index.
// Uses bitwise AND instead of modulo for performance (capacity must be power
// of two).
static inline iree_host_size_t iree_tokenizer_ring_to_physical(
    iree_host_size_t logical_position, iree_host_size_t capacity_mask) {
  return logical_position & capacity_mask;
}

// Returns the number of bytes currently in the ring buffer.
// This is the distance between read and write positions.
static inline iree_host_size_t iree_tokenizer_ring_used(
    const iree_tokenizer_encode_state_t* state) {
  return state->write_position - state->read_position;
}

// Returns the total available space in the ring buffer.
// This is capacity minus used bytes.
static inline iree_host_size_t iree_tokenizer_ring_available(
    const iree_tokenizer_encode_state_t* state) {
  iree_host_size_t capacity = state->capacity_mask + 1;
  return capacity - iree_tokenizer_ring_used(state);
}

// Returns the available ring space for writing. Writes may extend into the
// mirror region [capacity, 2*capacity) since the buffer is allocated 2x
// capacity. After writing, callers must call ring_fixup_write_wrap to copy
// any overflow from the mirror back to the buffer start.
static inline iree_host_size_t iree_tokenizer_ring_writable_space(
    const iree_tokenizer_encode_state_t* state) {
  return iree_tokenizer_ring_available(state);
}

// After writing |bytes_written| bytes starting at |physical_write_start|,
// copies any bytes that extended past the buffer capacity from the mirror
// region [capacity, capacity + overflow) back to the buffer start [0,
// overflow). This maintains the ring invariant that logical positions map
// correctly via (position & capacity_mask).
static inline void iree_tokenizer_ring_fixup_write_wrap(
    iree_tokenizer_encode_state_t* state, iree_host_size_t physical_write_start,
    iree_host_size_t bytes_written) {
  iree_host_size_t capacity = state->capacity_mask + 1;
  if (physical_write_start + bytes_written > capacity) {
    iree_host_size_t overflow = physical_write_start + bytes_written - capacity;
    memcpy(state->transform_buffer.data,
           state->transform_buffer.data + capacity, overflow);
  }
}

// Returns a contiguous view of data from segmenter_view_start to
// write_position.
//
// DOUBLE-BUFFER MODE (current implementation):
// If data spans the wrap boundary, the wrap-around portion [0, physical_write)
// is copied to the mirror region [capacity, capacity + physical_write) to
// provide contiguous access. The allocation must be 2x logical capacity.
//
// Future modes (compile-time selection, not yet implemented):
// - MEMMOVE: Shift data to buffer start instead of copying to mirror.
//   Trades 2x allocation for occasional data movement.
// - MAGIC: Use MMU page mapping for zero-copy contiguous access.
//   Requires platform-specific support, best for large persistent buffers.
//
// The returned view is valid until the next ring write operation.
static iree_string_view_t iree_tokenizer_ring_get_segment_view(
    iree_tokenizer_encode_state_t* state) {
  // Guard against invariant violation: segmenter_view_start must not exceed
  // write_position. Check BEFORE subtraction to avoid underflow.
  if (state->segmenter_view_start >= state->write_position) {
    return iree_make_string_view(NULL, 0);
  }
  iree_host_size_t total_length =
      state->write_position - state->segmenter_view_start;

  iree_host_size_t capacity = state->capacity_mask + 1;
  iree_host_size_t physical_segment = iree_tokenizer_ring_to_physical(
      state->segmenter_view_start, state->capacity_mask);
  iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
      state->write_position, state->capacity_mask);

  // If data doesn't wrap (write is strictly after segment), return direct view.
  if (physical_write > physical_segment) {
    return iree_make_string_view(
        (const char*)state->transform_buffer.data + physical_segment,
        total_length);
  }

  // Data wraps around buffer end. Copy wrap-around portion to mirror region
  // to provide contiguous access. Mirror region is at [capacity, 2*capacity).
  //
  // Before: [wrap_data...][...main_data]
  //         ^             ^
  //         0             physical_segment
  //
  // After:  [wrap_data...][...main_data][wrap_copy...]
  //         ^             ^             ^
  //         0             physical_segment  capacity (mirror)
  //
  // View starts at physical_segment and extends into mirror region.
  iree_host_size_t wrap_length = physical_write;
  if (wrap_length > 0) {
    memcpy(state->transform_buffer.data + capacity,
           state->transform_buffer.data, wrap_length);
  }

  return iree_make_string_view(
      (const char*)state->transform_buffer.data + physical_segment,
      total_length);
}

//===----------------------------------------------------------------------===//
// Builder Implementation
//===----------------------------------------------------------------------===//

void iree_tokenizer_builder_initialize(iree_allocator_t allocator,
                                       iree_tokenizer_builder_t* out_builder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->allocator = allocator;
  out_builder->string_batch_size = IREE_TOKENIZER_DEFAULT_STRING_BATCH_SIZE;
  IREE_TRACE_ZONE_END(z0);
}

void iree_tokenizer_builder_deinitialize(iree_tokenizer_builder_t* builder) {
  if (!builder) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Free any components still owned by builder.
  // After successful build(), components are transferred to tokenizer,
  // so this becomes a no-op. Only frees on error path.
  iree_tokenizer_normalizer_free(builder->normalizer);
  iree_tokenizer_segmenter_free(builder->segmenter);
  // Free model BEFORE vocab since model references but doesn't own vocab.
  iree_tokenizer_model_free(builder->model);
  iree_tokenizer_decoder_free(builder->decoder);
  iree_tokenizer_postprocessor_deinitialize(&builder->postprocessor);
  iree_tokenizer_special_tokens_deinitialize(&builder->special_tokens);
  iree_tokenizer_special_tokens_deinitialize(
      &builder->special_tokens_post_norm);
  iree_tokenizer_vocab_free(builder->vocab);

  memset(builder, 0, sizeof(*builder));
  IREE_TRACE_ZONE_END(z0);
}

void iree_tokenizer_builder_set_normalizer(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing normalizer if any.
  iree_tokenizer_normalizer_free(builder->normalizer);
  builder->normalizer = normalizer;
}

void iree_tokenizer_builder_set_segmenter(
    iree_tokenizer_builder_t* builder, iree_tokenizer_segmenter_t* segmenter) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing segmenter if any.
  iree_tokenizer_segmenter_free(builder->segmenter);
  builder->segmenter = segmenter;
}

void iree_tokenizer_builder_set_model(iree_tokenizer_builder_t* builder,
                                      iree_tokenizer_model_t* model) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing model if any.
  iree_tokenizer_model_free(builder->model);
  builder->model = model;
}

void iree_tokenizer_builder_set_decoder(iree_tokenizer_builder_t* builder,
                                        iree_tokenizer_decoder_t* decoder) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing decoder if any.
  iree_tokenizer_decoder_free(builder->decoder);
  builder->decoder = decoder;
}

void iree_tokenizer_builder_set_postprocessor(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_postprocessor_t postprocessor) {
  IREE_ASSERT_ARGUMENT(builder);
  iree_tokenizer_postprocessor_deinitialize(&builder->postprocessor);
  builder->postprocessor = postprocessor;
}

void iree_tokenizer_builder_set_special_tokens(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_special_tokens_t* special_tokens) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(special_tokens);
  iree_tokenizer_special_tokens_deinitialize(&builder->special_tokens);
  // Move ownership: copy struct and clear source.
  builder->special_tokens = *special_tokens;
  memset(special_tokens, 0, sizeof(*special_tokens));
}

void iree_tokenizer_builder_set_special_tokens_post_norm(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_special_tokens_t* special_tokens) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(special_tokens);
  iree_tokenizer_special_tokens_deinitialize(
      &builder->special_tokens_post_norm);
  // Move ownership: copy struct and clear source.
  builder->special_tokens_post_norm = *special_tokens;
  memset(special_tokens, 0, sizeof(*special_tokens));
}

void iree_tokenizer_builder_set_vocab(iree_tokenizer_builder_t* builder,
                                      iree_tokenizer_vocab_t* vocab) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing vocab if any.
  iree_tokenizer_vocab_free(builder->vocab);
  builder->vocab = vocab;
}

void iree_tokenizer_builder_set_string_batch_size(
    iree_tokenizer_builder_t* builder, iree_host_size_t batch_size) {
  IREE_ASSERT_ARGUMENT(builder);
  builder->string_batch_size = batch_size;
}

//===----------------------------------------------------------------------===//
// Pre-decode Build
//===----------------------------------------------------------------------===//

// Decodes a single token through the decoder chain, returning its decoded bytes
// in work_buffer[0..out_decoded_length). For position-sensitive decoders, this
// produces the "rest" form (as if the token is NOT the first in the stream).
static iree_status_t iree_tokenizer_pre_decode_one_token(
    const iree_tokenizer_decoder_t* decoder, iree_string_view_t token_text,
    bool position_sensitive, void* state_storage, char* work_buffer,
    iree_host_size_t work_buffer_size, iree_host_size_t* out_decoded_length) {
  *out_decoded_length = 0;

  // Initialize decoder state.
  iree_tokenizer_decoder_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_decoder_state_initialize(decoder, state_storage, &state));

  // For position-sensitive decoders, feed a dummy token to advance past
  // first-token handling. This ensures we get the "rest" form (with leading
  // space where applicable).
  if (position_sensitive) {
    iree_string_view_t dummy = iree_make_cstring_view("a");
    iree_tokenizer_string_list_t dummy_list =
        iree_tokenizer_make_string_list(&dummy, 1);
    iree_mutable_string_view_t discard_output =
        iree_make_mutable_string_view(work_buffer, work_buffer_size);
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    iree_status_t status = iree_tokenizer_decoder_state_process(
        state, dummy_list, discard_output, &consumed, &written);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_decoder_state_deinitialize(state);
      return status;
    }
  }

  iree_host_size_t total_written = 0;

  // Feed the real token.
  if (token_text.size > 0) {
    iree_tokenizer_string_list_t token_list =
        iree_tokenizer_make_string_list(&token_text, 1);
    iree_mutable_string_view_t output =
        iree_make_mutable_string_view(work_buffer, work_buffer_size);
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    iree_status_t status = iree_tokenizer_decoder_state_process(
        state, token_list, output, &consumed, &written);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_decoder_state_deinitialize(state);
      return status;
    }
    total_written = written;
  }

  // Finalize to flush any pending data (e.g., ByteLevel incomplete UTF-8).
  {
    iree_mutable_string_view_t finalize_output = iree_make_mutable_string_view(
        work_buffer + total_written, work_buffer_size - total_written);
    iree_host_size_t finalize_written = 0;
    iree_status_t status = iree_tokenizer_decoder_state_finalize(
        state, finalize_output, &finalize_written);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_decoder_state_deinitialize(state);
      return status;
    }
    total_written += finalize_written;
  }

  iree_tokenizer_decoder_state_deinitialize(state);
  *out_decoded_length = total_written;
  return iree_ok_status();
}

// Builds the pre-decoded token string table. Called once at tokenizer
// construction time. If the decoder supports pre-decode (STATELESS or
// STATELESS_EXCEPT_BYTE_TOKENS capability), this pre-computes decoded strings
// for all vocab tokens, enabling O(1) memcpy-based decode at runtime.
//
// For STATELESS_EXCEPT_BYTE_TOKENS (ByteFallback models like Gemma):
// Non-byte tokens are pre-decoded through the full chain as normal.
// Byte tokens (<0xHH>) store their single raw byte value (1 byte each).
// At decode time, byte tokens are handled by an inline accumulator that
// validates UTF-8 sequences, while non-byte tokens use the memcpy fast path.
static iree_status_t iree_tokenizer_build_pre_decoded(
    iree_tokenizer_t* tokenizer) {
  if (!tokenizer->decoder || !tokenizer->vocab) return iree_ok_status();

  iree_tokenizer_decoder_capability_t capabilities =
      iree_tokenizer_decoder_capabilities(tokenizer->decoder);

  bool has_byte_fallback =
      !!(capabilities &
         IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS_EXCEPT_BYTE_TOKENS);
  bool is_stateless =
      !!(capabilities & IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS);

  if (!is_stateless && !has_byte_fallback) {
    return iree_ok_status();  // Not pre-decodable.
  }

  bool position_sensitive =
      !!(capabilities & IREE_TOKENIZER_DECODER_CAPABILITY_POSITION_SENSITIVE);

  iree_host_size_t vocab_capacity =
      iree_tokenizer_vocab_capacity(tokenizer->vocab);
  if (vocab_capacity == 0) return iree_ok_status();

  iree_host_size_t max_token_length =
      iree_tokenizer_vocab_max_token_length(tokenizer->vocab);

  // Work buffer: max decoded length per token. WordPiece can add 1 byte
  // (space prefix), ByteLevel finalize can add 3 bytes (U+FFFD).
  iree_host_size_t work_buffer_size = max_token_length + 4;
  iree_host_size_t decoder_state_size =
      iree_tokenizer_decoder_state_size(tokenizer->decoder);

  // Allocate temporary working memory: [state_storage][work_buffer]
  iree_host_size_t temp_size = decoder_state_size + work_buffer_size;
  iree_host_size_t offsets_size = 0;
  if (!iree_host_size_checked_mul(vocab_capacity + 1, sizeof(uint32_t),
                                  &offsets_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "offsets array size overflow");
  }
  void* temp_storage = NULL;
  void* slab = NULL;
  iree_status_t status =
      iree_allocator_malloc(tokenizer->allocator, temp_size, &temp_storage);

  // Detect byte token ID range for ByteFallback models.
  // Byte tokens (<0x00> through <0xFF>) are nearly contiguous in SentencePiece
  // vocabs but may have small gaps (e.g., Gemma puts a literal tab at ID 226
  // inside its byte token range 217-472). We use a 256-bit bitmap to handle
  // gaps efficiently at decode time.
  int32_t byte_token_first_id = -1;
  int32_t byte_token_last_id = -1;
  uint8_t byte_token_bitmap[32];
  memset(byte_token_bitmap, 0, sizeof(byte_token_bitmap));

  if (iree_status_is_ok(status) && has_byte_fallback) {
    // Scan vocab to find all byte tokens.
    int byte_token_count = 0;
    for (iree_host_size_t id = 0; id < vocab_capacity; ++id) {
      iree_string_view_t token_text =
          iree_tokenizer_vocab_token_text(tokenizer->vocab, (int32_t)id);
      uint8_t byte_value;
      if (iree_tokenizer_decoder_byte_fallback_parse_byte_token(token_text,
                                                                &byte_value)) {
        if (byte_token_first_id < 0) byte_token_first_id = (int32_t)id;
        byte_token_last_id = (int32_t)id;
        ++byte_token_count;
      }
    }

    if (byte_token_count > 0 && byte_token_count <= 256) {
      // Verify the range fits in our 256-bit bitmap.
      int32_t range_span = byte_token_last_id - byte_token_first_id + 1;
      if (range_span > 256) {
        iree_allocator_free(tokenizer->allocator, temp_storage);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "byte token ID range too wide: %d tokens spanning %" PRId32
            " to %" PRId32 " (max span 256)",
            byte_token_count, byte_token_first_id, byte_token_last_id);
      }

      // Build bitmap: re-scan to mark byte token positions.
      for (iree_host_size_t id = (iree_host_size_t)byte_token_first_id;
           id <= (iree_host_size_t)byte_token_last_id; ++id) {
        iree_string_view_t token_text =
            iree_tokenizer_vocab_token_text(tokenizer->vocab, (int32_t)id);
        uint8_t byte_value;
        if (iree_tokenizer_decoder_byte_fallback_parse_byte_token(
                token_text, &byte_value)) {
          int32_t bit_index = (int32_t)id - byte_token_first_id;
          byte_token_bitmap[bit_index / 8] |= (1u << (bit_index % 8));
        }
      }
    } else if (byte_token_count == 0) {
      // No byte tokens found despite ByteFallback capability — the decoder
      // will still function correctly, it just has nothing to accumulate.
      // Treat as fully stateless since there are no byte tokens to handle.
      has_byte_fallback = false;
    }
  }

  if (iree_status_is_ok(status)) {
    void* state_storage = temp_storage;
    char* work_buffer = (char*)temp_storage + decoder_state_size;

    // Pass 1: compute total decoded size.
    iree_host_size_t total_data_size = 0;
    for (iree_host_size_t id = 0;
         id < vocab_capacity && iree_status_is_ok(status); ++id) {
      iree_string_view_t token_text =
          iree_tokenizer_vocab_token_text(tokenizer->vocab, (int32_t)id);

      // Byte tokens store 1 raw byte instead of going through the decoder.
      uint8_t byte_value;
      if (has_byte_fallback &&
          iree_tokenizer_decoder_byte_fallback_parse_byte_token(token_text,
                                                                &byte_value)) {
        total_data_size += 1;
        continue;
      }

      iree_host_size_t decoded_length = 0;
      status = iree_tokenizer_pre_decode_one_token(
          tokenizer->decoder, token_text, position_sensitive, state_storage,
          work_buffer, work_buffer_size, &decoded_length);
      total_data_size += decoded_length;
    }

    // Allocate slab: [offsets: (vocab_capacity + 1) * sizeof(uint32_t)][data]
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(tokenizer->allocator,
                                     offsets_size + total_data_size, &slab);
    }

    // Pass 2: populate data blob and offsets.
    if (iree_status_is_ok(status)) {
      uint32_t* offsets = (uint32_t*)slab;
      uint8_t* data = (uint8_t*)slab + offsets_size;
      uint32_t data_position = 0;
      for (iree_host_size_t id = 0;
           id < vocab_capacity && iree_status_is_ok(status); ++id) {
        offsets[id] = data_position;
        iree_string_view_t token_text =
            iree_tokenizer_vocab_token_text(tokenizer->vocab, (int32_t)id);

        // Byte tokens: store the single raw byte value directly.
        uint8_t byte_value;
        if (has_byte_fallback &&
            iree_tokenizer_decoder_byte_fallback_parse_byte_token(
                token_text, &byte_value)) {
          data[data_position] = byte_value;
          data_position += 1;
          continue;
        }

        iree_host_size_t decoded_length = 0;
        status = iree_tokenizer_pre_decode_one_token(
            tokenizer->decoder, token_text, position_sensitive, state_storage,
            work_buffer, work_buffer_size, &decoded_length);
        if (iree_status_is_ok(status)) {
          memcpy(data + data_position, work_buffer, decoded_length);
          data_position += (uint32_t)decoded_length;
        }
      }
      offsets[vocab_capacity] = data_position;
    }
  }

  iree_allocator_free(tokenizer->allocator, temp_storage);

  if (iree_status_is_ok(status)) {
    tokenizer->pre_decoded.slab = slab;
    tokenizer->pre_decoded.offsets = (uint32_t*)slab;
    tokenizer->pre_decoded.data = (uint8_t*)slab + offsets_size;
    tokenizer->pre_decoded.position_sensitive = position_sensitive;
    tokenizer->pre_decoded.has_byte_fallback = has_byte_fallback;
    tokenizer->pre_decoded.byte_token_first_id = byte_token_first_id;
    tokenizer->pre_decoded.byte_token_last_id = byte_token_last_id;
    memcpy(tokenizer->pre_decoded.byte_token_bitmap, byte_token_bitmap,
           sizeof(byte_token_bitmap));
  } else {
    iree_allocator_free(tokenizer->allocator, slab);
  }

  return status;
}

iree_status_t iree_tokenizer_builder_build(iree_tokenizer_builder_t* builder,
                                           iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;

  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate required components.
  // Note: segmenter is optional - NULL means passthrough (entire input as one
  // segment). This is used by tokenizers with null pre_tokenizer (TinyLlama,
  // Phi-3, etc.).
  iree_status_t status = iree_ok_status();
  if (!builder->model) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "model is required");
  }
  if (iree_status_is_ok(status) && builder->string_batch_size == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "string_batch_size must be > 0");
  }

  // Allocate tokenizer struct.
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(builder->allocator, sizeof(*tokenizer),
                                   (void**)&tokenizer);
  }

  if (iree_status_is_ok(status)) {
    tokenizer->allocator = builder->allocator;
    tokenizer->segment_batch_size = IREE_TOKENIZER_DEFAULT_SEGMENT_BATCH_SIZE;
    tokenizer->string_batch_size = builder->string_batch_size;

    // Transfer ownership of components from builder to tokenizer.
    tokenizer->normalizer = builder->normalizer;
    tokenizer->segmenter = builder->segmenter;
    tokenizer->model = builder->model;
    tokenizer->decoder = builder->decoder;
    tokenizer->postprocessor = builder->postprocessor;
    tokenizer->special_tokens = builder->special_tokens;
    tokenizer->special_tokens_post_norm = builder->special_tokens_post_norm;
    tokenizer->vocab = builder->vocab;

    // Clear builder (ownership transferred).
    builder->normalizer = NULL;
    builder->segmenter = NULL;
    builder->model = NULL;
    builder->decoder = NULL;
    memset(&builder->postprocessor, 0, sizeof(builder->postprocessor));
    memset(&builder->special_tokens, 0, sizeof(builder->special_tokens));
    memset(&builder->special_tokens_post_norm, 0,
           sizeof(builder->special_tokens_post_norm));
    builder->vocab = NULL;

    // Build pre-decoded token table for fast-path decode.
    status = iree_tokenizer_build_pre_decoded(tokenizer);
  }

  if (iree_status_is_ok(status)) {
    *out_tokenizer = tokenizer;
  } else {
    iree_tokenizer_free(tokenizer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

void iree_tokenizer_free(iree_tokenizer_t* tokenizer) {
  if (!tokenizer) return;

  iree_allocator_t allocator = tokenizer->allocator;

  // Free pre-decoded table (single slab covers offsets + data).
  iree_allocator_free(allocator, tokenizer->pre_decoded.slab);

  // Free owned components.
  iree_tokenizer_normalizer_free(tokenizer->normalizer);
  iree_tokenizer_segmenter_free(tokenizer->segmenter);
  // Free model BEFORE vocab since model references but doesn't own vocab.
  iree_tokenizer_model_free(tokenizer->model);
  iree_tokenizer_decoder_free(tokenizer->decoder);
  iree_tokenizer_postprocessor_deinitialize(&tokenizer->postprocessor);
  iree_tokenizer_special_tokens_deinitialize(&tokenizer->special_tokens);
  iree_tokenizer_special_tokens_deinitialize(
      &tokenizer->special_tokens_post_norm);
  iree_tokenizer_vocab_free(tokenizer->vocab);

  iree_allocator_free(allocator, tokenizer);
}

const iree_tokenizer_vocab_t* iree_tokenizer_vocab(
    const iree_tokenizer_t* tokenizer) {
  return tokenizer->vocab;
}

iree_string_view_t iree_tokenizer_model_type_name(
    const iree_tokenizer_t* tokenizer) {
  return tokenizer->model->type_name;
}

//===----------------------------------------------------------------------===//
// Batch Encode/Decode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_encode(const iree_tokenizer_t* tokenizer,
                                    iree_string_view_t text,
                                    iree_tokenizer_encode_flags_t flags,
                                    iree_tokenizer_token_output_t output,
                                    iree_allocator_t allocator,
                                    iree_host_size_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(output.token_ids);
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_token_count = 0;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate combined buffer for state + transform.
  iree_host_size_t state_size = 0;
  iree_status_t status =
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size);

  // One-shot encoding needs a transform buffer proportional to the input since
  // pre-tokenizers with infrequent splits (e.g., DeepSeek-V3's compound
  // Sequence pre-tokenizer on English text) can produce segments spanning tens
  // of KB, exceeding the streaming-optimized 64KB cap. The streaming
  // recommended_size() caps at 64KB for bounded memory; one-shot is uncapped
  // since we already allocate O(text_size) for output tokens.
  iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_oneshot_size(text.size);
  iree_host_size_t total_size = 0;
  iree_host_size_t transform_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        state_size, &total_size,
        IREE_STRUCT_ARRAY_FIELD(transform_size, 1, uint8_t, &transform_offset));
  }

  uint8_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, total_size, (void**)&buffer);
  }

  // Initialize and run pipeline. AT_INPUT_START is always set for fresh
  // encodes. TRACK_OFFSETS is inferred from the output having an offsets
  // buffer.
  iree_tokenizer_encode_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    iree_tokenizer_encode_flags_t effective_flags =
        flags | IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
    if (output.token_offsets) {
      effective_flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
    }
    status = iree_tokenizer_encode_state_initialize(
        tokenizer, iree_make_byte_span(buffer, state_size),
        iree_make_byte_span(buffer + transform_offset, transform_size),
        iree_tokenizer_offset_run_list_empty(), effective_flags, &state);
  }

  // Feed all input, collecting tokens.
  iree_host_size_t total_tokens = 0;
  while (iree_status_is_ok(status) && text.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t tokens_written = 0;
    iree_tokenizer_token_output_t sub_output = {
        .capacity = output.capacity - total_tokens,
        .token_ids = &output.token_ids[total_tokens],
        .token_offsets =
            output.token_offsets ? &output.token_offsets[total_tokens] : NULL,
        .type_ids = output.type_ids ? &output.type_ids[total_tokens] : NULL,
    };
    status = iree_tokenizer_encode_state_feed(state, text, sub_output,
                                              &bytes_consumed, &tokens_written);
    total_tokens += tokens_written;
    text.data += bytes_consumed;
    text.size -= bytes_consumed;

    // Handle no-progress case: normalizer waiting for lookahead at stream end.
    // Since this is batch encode (all input provided at once), remaining bytes
    // after no-progress are definitively at end-of-stream. Force them through
    // the normalizer with SEGMENT_END flag.
    //
    // First check for incomplete UTF-8 at end of input. The normalizer contract
    // requires input to end on codepoint boundaries. If the remaining bytes are
    // an incomplete sequence, this is a caller error (batch API expects
    // complete UTF-8 input). Return INVALID_ARGUMENT rather than corrupting
    // output.
    if (iree_status_is_ok(status) && bytes_consumed == 0 && text.size > 0) {
      iree_host_size_t incomplete =
          iree_unicode_utf8_incomplete_tail_length(text.data, text.size);
      if (incomplete == text.size) {
        // All remaining bytes are an incomplete UTF-8 sequence.
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "batch encode input ends with incomplete UTF-8 sequence "
            "(%" PRIhsz " trailing bytes)",
            text.size);
      } else if (state->normalizer_state) {
        // Truncate to complete codepoints before feeding to normalizer.
        iree_string_view_t safe_text = {text.data, text.size - incomplete};
        iree_host_size_t write_space =
            iree_tokenizer_ring_writable_space(state);
        if (write_space > 0) {
          iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
              state->write_position, state->capacity_mask);
          iree_mutable_string_view_t norm_output = {
              .data = (char*)state->transform_buffer.data + physical_write,
              .size = write_space,
          };
          iree_host_size_t norm_consumed = 0;
          iree_host_size_t norm_written = 0;
          status = iree_tokenizer_normalizer_state_process(
              state->normalizer_state, safe_text, norm_output,
              IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END, &norm_consumed,
              &norm_written);
          iree_tokenizer_ring_fixup_write_wrap(state, physical_write,
                                               norm_written);
          state->write_position += norm_written;
          text.data += norm_consumed;
          text.size -= norm_consumed;
        }
      }
    }
  }

  // Finalize to flush remaining tokens.
  if (iree_status_is_ok(status)) {
    iree_host_size_t final_tokens = 0;
    iree_tokenizer_token_output_t final_output = {
        .capacity = output.capacity - total_tokens,
        .token_ids = &output.token_ids[total_tokens],
        .token_offsets =
            output.token_offsets ? &output.token_offsets[total_tokens] : NULL,
        .type_ids = output.type_ids ? &output.type_ids[total_tokens] : NULL,
    };
    status = iree_tokenizer_encode_state_finalize(state, final_output,
                                                  &final_tokens);
    total_tokens += final_tokens;
  }

  if (iree_status_is_ok(status)) {
    *out_token_count = total_tokens;
  }

  // Cleanup.
  iree_tokenizer_encode_state_deinitialize(state);
  iree_allocator_free(allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_decode(const iree_tokenizer_t* tokenizer,
                                    iree_tokenizer_token_id_list_t tokens,
                                    iree_tokenizer_decode_flags_t flags,
                                    iree_mutable_string_view_t text_output,
                                    iree_allocator_t allocator,
                                    iree_host_size_t* out_text_length) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(tokens.values || tokens.count == 0);
  IREE_ASSERT_ARGUMENT(text_output.data);
  IREE_ASSERT_ARGUMENT(out_text_length);
  *out_text_length = 0;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate state size.
  iree_host_size_t state_size = 0;
  iree_status_t status =
      iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size);

  // Allocate state storage.
  uint8_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, state_size, (void**)&buffer);
  }

  // Initialize decode state.
  iree_tokenizer_decode_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_decode_state_initialize(
        tokenizer, flags, iree_make_byte_span(buffer, state_size), &state);
  }

  // Feed all tokens, collecting text.
  iree_host_size_t total_written = 0;
  iree_mutable_string_view_t remaining_output = text_output;
  while (iree_status_is_ok(status) && tokens.count > 0) {
    iree_host_size_t tokens_consumed = 0;
    iree_host_size_t text_written = 0;
    status = iree_tokenizer_decode_state_feed(state, tokens, remaining_output,
                                              &tokens_consumed, &text_written);
    total_written += text_written;
    remaining_output.data += text_written;
    remaining_output.size -= text_written;
    tokens.values += tokens_consumed;
    tokens.count -= tokens_consumed;
    // Zero progress means the output buffer is exhausted. This occurs when:
    //  - Pre-decoded path: first token's text exceeds remaining capacity.
    //  - Pipeline path: decoder can't write (output too small) AND string
    //    buffer is full (no room for Stage 2 to consume tokens).
    // The pump loop's made_progress flag handles all internal "needs more
    // pumping" cases, so 0/0 at this level is always genuine exhaustion.
    if (tokens_consumed == 0 && text_written == 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "decode output buffer full");
    }
  }

  // Finalize to flush remaining buffered data.
  if (iree_status_is_ok(status)) {
    iree_host_size_t final_written = 0;
    status = iree_tokenizer_decode_state_finalize(state, remaining_output,
                                                  &final_written);
    total_written += final_written;
  }

  if (iree_status_is_ok(status)) {
    *out_text_length = total_written;
  }

  // Cleanup.
  iree_tokenizer_decode_state_deinitialize(state);
  iree_allocator_free(allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Multi-Item Batch Encode/Decode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_encode_batch(
    const iree_tokenizer_t* tokenizer,
    iree_tokenizer_encode_batch_item_t* items, iree_host_size_t item_count,
    iree_tokenizer_encode_flags_t flags, iree_byte_span_t state_storage,
    iree_byte_span_t transform_buffer,
    iree_tokenizer_offset_run_list_t offset_runs) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(items || item_count == 0);

  // Initialize all output counts to zero.
  for (iree_host_size_t i = 0; i < item_count; ++i) {
    items[i].out_token_count = 0;
  }

  iree_status_t status = iree_ok_status();
  iree_tokenizer_encode_state_t* state = NULL;

  // Process each item, reusing state storage across items. Loop exits early on
  // error; prior items retain their results.
  for (iree_host_size_t i = 0; i < item_count && iree_status_is_ok(status);
       ++i) {
    iree_tokenizer_encode_batch_item_t* item = &items[i];

    // AT_INPUT_START always set for fresh encodes. TRACK_OFFSETS inferred
    // per-item from whether the output has an offsets buffer.
    iree_tokenizer_encode_flags_t effective_flags =
        flags | IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
    if (item->output.token_offsets) {
      effective_flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
    }

    // First item: full initialize. Subsequent items: lightweight reset.
    if (state == NULL) {
      status = iree_tokenizer_encode_state_initialize(
          tokenizer, state_storage, transform_buffer, offset_runs,
          effective_flags, &state);
    } else {
      iree_tokenizer_encode_state_reset(state, effective_flags);
    }

    // Feed all text, collecting tokens.
    iree_string_view_t text = item->text;
    iree_host_size_t total_tokens = 0;

    while (iree_status_is_ok(status) && text.size > 0) {
      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t tokens_written = 0;
      iree_tokenizer_token_output_t sub_output = {
          .capacity = item->output.capacity - total_tokens,
          .token_ids = &item->output.token_ids[total_tokens],
          .token_offsets = item->output.token_offsets
                               ? &item->output.token_offsets[total_tokens]
                               : NULL,
          .type_ids = item->output.type_ids
                          ? &item->output.type_ids[total_tokens]
                          : NULL,
      };
      status = iree_tokenizer_encode_state_feed(
          state, text, sub_output, &bytes_consumed, &tokens_written);
      total_tokens += tokens_written;
      text.data += bytes_consumed;
      text.size -= bytes_consumed;
    }

    // Finalize to flush remaining tokens.
    if (iree_status_is_ok(status)) {
      iree_host_size_t final_tokens = 0;
      iree_tokenizer_token_output_t final_output = {
          .capacity = item->output.capacity - total_tokens,
          .token_ids = &item->output.token_ids[total_tokens],
          .token_offsets = item->output.token_offsets
                               ? &item->output.token_offsets[total_tokens]
                               : NULL,
          .type_ids = item->output.type_ids
                          ? &item->output.type_ids[total_tokens]
                          : NULL,
      };
      status = iree_tokenizer_encode_state_finalize(state, final_output,
                                                    &final_tokens);
      total_tokens += final_tokens;
    }

    item->out_token_count = total_tokens;
  }

  iree_tokenizer_encode_state_deinitialize(state);
  return status;
}

iree_status_t iree_tokenizer_decode_batch(
    const iree_tokenizer_t* tokenizer,
    iree_tokenizer_decode_batch_item_t* items, iree_host_size_t item_count,
    iree_tokenizer_decode_flags_t flags, iree_byte_span_t state_storage) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(items || item_count == 0);

  // Initialize all output lengths to zero.
  for (iree_host_size_t i = 0; i < item_count; ++i) {
    items[i].out_text_length = 0;
  }

  // Process each item, reusing state storage across items.
  for (iree_host_size_t i = 0; i < item_count; ++i) {
    iree_tokenizer_decode_batch_item_t* item = &items[i];

    // Initialize state for this item (reuses storage from previous item).
    iree_tokenizer_decode_state_t* state = NULL;
    IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_initialize(
        tokenizer, flags, state_storage, &state));

    // Feed all tokens, collecting text.
    iree_tokenizer_token_id_list_t tokens = item->tokens;
    iree_host_size_t total_written = 0;
    iree_mutable_string_view_t remaining_output = item->text_output;
    iree_status_t status = iree_ok_status();

    while (iree_status_is_ok(status) && tokens.count > 0) {
      iree_host_size_t tokens_consumed = 0;
      iree_host_size_t text_written = 0;
      status = iree_tokenizer_decode_state_feed(
          state, tokens, remaining_output, &tokens_consumed, &text_written);
      total_written += text_written;
      remaining_output.data += text_written;
      remaining_output.size -= text_written;
      tokens.values += tokens_consumed;
      tokens.count -= tokens_consumed;
      // Zero progress = output buffer exhausted (see one-shot decode comment).
      if (tokens_consumed == 0 && text_written == 0) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "decode output buffer full");
      }
    }

    // Finalize to flush remaining buffered data.
    if (iree_status_is_ok(status)) {
      iree_host_size_t final_written = 0;
      status = iree_tokenizer_decode_state_finalize(state, remaining_output,
                                                    &final_written);
      total_written += final_written;
    }

    item->out_text_length = total_written;
    iree_tokenizer_decode_state_deinitialize(state);

    if (!iree_status_is_ok(status)) {
      return status;  // Stop at first failure, prior items retain results.
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Streaming Encode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_encode_state_calculate_size(
    const iree_tokenizer_t* tokenizer, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;

  iree_host_size_t normalizer_state_size =
      tokenizer->normalizer
          ? iree_tokenizer_normalizer_state_size(tokenizer->normalizer)
          : 0;
  iree_host_size_t segmenter_state_size =
      tokenizer->segmenter
          ? iree_tokenizer_segmenter_state_size(tokenizer->segmenter)
          : 0;
  iree_host_size_t model_state_size =
      tokenizer->model ? iree_tokenizer_model_state_size(tokenizer->model) : 0;

  return IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_encode_state_t), out_size,
      IREE_STRUCT_FIELD_ALIGNED(normalizer_state_size, uint8_t,
                                iree_max_align_t, NULL),
      IREE_STRUCT_FIELD_ALIGNED(segmenter_state_size, uint8_t, iree_max_align_t,
                                NULL),
      IREE_STRUCT_FIELD_ALIGNED(model_state_size, uint8_t, iree_max_align_t,
                                NULL),
      IREE_STRUCT_FIELD(tokenizer->segment_batch_size, iree_tokenizer_segment_t,
                        NULL));
}

iree_status_t iree_tokenizer_encode_state_initialize(
    const iree_tokenizer_t* tokenizer, iree_byte_span_t state_storage,
    iree_byte_span_t transform_buffer,
    iree_tokenizer_offset_run_list_t offset_runs,
    iree_tokenizer_encode_flags_t flags,
    iree_tokenizer_encode_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(state_storage.data);
  IREE_ASSERT_ARGUMENT(transform_buffer.data);
  IREE_ASSERT_ARGUMENT(out_state);

  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate layout with offsets. Must match calculate_size exactly.
  iree_host_size_t normalizer_state_size =
      tokenizer->normalizer
          ? iree_tokenizer_normalizer_state_size(tokenizer->normalizer)
          : 0;
  iree_host_size_t segmenter_state_size =
      tokenizer->segmenter
          ? iree_tokenizer_segmenter_state_size(tokenizer->segmenter)
          : 0;
  iree_host_size_t model_state_size =
      tokenizer->model ? iree_tokenizer_model_state_size(tokenizer->model) : 0;

  iree_host_size_t total_size = 0;
  iree_host_size_t normalizer_offset = 0;
  iree_host_size_t segmenter_offset = 0;
  iree_host_size_t model_offset = 0;
  iree_host_size_t segments_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_encode_state_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(normalizer_state_size, uint8_t,
                                iree_max_align_t, &normalizer_offset),
      IREE_STRUCT_FIELD_ALIGNED(segmenter_state_size, uint8_t, iree_max_align_t,
                                &segmenter_offset),
      IREE_STRUCT_FIELD_ALIGNED(model_state_size, uint8_t, iree_max_align_t,
                                &model_offset),
      IREE_STRUCT_FIELD(tokenizer->segment_batch_size, iree_tokenizer_segment_t,
                        &segments_offset));

  if (iree_status_is_ok(status) && state_storage.data_length < total_size) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "state_storage too small: need %" PRIhsz
                              " bytes, got %" PRIhsz,
                              total_size, state_storage.data_length);
  }

  // Validate transform buffer allocation for double-buffer mode.
  // Allocation must be a power of two. Logical capacity is half the allocation:
  // first half is the working buffer, second half is the mirror region for
  // contiguous wrap-spanning access (see
  // iree_tokenizer_ring_get_segment_view).
  //
  // Minimum allocation is 2 * IREE_UNICODE_UTF8_MAX_BYTE_LENGTH = 8 bytes,
  // ensuring the logical capacity can hold at least one complete UTF-8
  // codepoint. In practice, 8KB-16KB is recommended for streaming.
  iree_host_size_t allocation = transform_buffer.data_length;
  iree_host_size_t min_allocation = 2 * IREE_UNICODE_UTF8_MAX_BYTE_LENGTH;
  if (iree_status_is_ok(status) &&
      (allocation < min_allocation ||
       !iree_host_size_is_power_of_two(allocation))) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transform buffer allocation must be a power of two >= %" PRIhsz
        ", got %" PRIhsz,
        min_allocation, allocation);
  }
  iree_host_size_t capacity = allocation / 2;

  // Validate buffer is large enough for the frozen token theorem to apply.
  // BPE's streaming mode freezes tokens when: current_byte > token_end +
  // max_token_length - 1. With capacity <= max_token_length, tokens at the
  // start of the buffer can never freeze, causing deadlock when the ring fills.
  // Require capacity > max_token_length (strictly greater) to guarantee
  // progress.
  iree_host_size_t max_token_length =
      iree_tokenizer_vocab_max_token_length(tokenizer->vocab);
  if (iree_status_is_ok(status) && capacity <= max_token_length) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transform buffer too small for this tokenizer: logical capacity "
        "%" PRIhsz " must be > max_token_length %" PRIhsz
        " (need physical allocation > %" PRIhsz ")",
        capacity, max_token_length, max_token_length * 2);
  }

  // Initialize state struct at base of storage.
  iree_tokenizer_encode_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    uint8_t* base = (uint8_t*)state_storage.data;
    state = (iree_tokenizer_encode_state_t*)base;
    memset(state, 0, sizeof(*state));

    state->tokenizer = tokenizer;
    state->transform_buffer = transform_buffer;
    state->offset_runs = offset_runs;
    state->flags = flags;
    state->capacity_mask = capacity - 1;
    state->read_position = 0;
    state->write_position = 0;
    state->segmenter_view_start = 0;
    state->segment_count = 0;
    state->segments_consumed = 0;
  }

  // Initialize normalizer state at computed offset.
  if (iree_status_is_ok(status) && tokenizer->normalizer) {
    uint8_t* base = (uint8_t*)state_storage.data;
    status = iree_tokenizer_normalizer_state_initialize(
        tokenizer->normalizer, base + normalizer_offset,
        &state->normalizer_state);
  }

  // Initialize segmenter state at computed offset.
  if (iree_status_is_ok(status) && tokenizer->segmenter) {
    uint8_t* base = (uint8_t*)state_storage.data;
    status = iree_tokenizer_segmenter_state_initialize(
        tokenizer->segmenter, base + segmenter_offset, &state->segmenter_state);
  }

  // Initialize model state at computed offset.
  if (iree_status_is_ok(status) && tokenizer->model) {
    uint8_t* base = (uint8_t*)state_storage.data;
    status = iree_tokenizer_model_state_initialize(
        tokenizer->model, base + model_offset, &state->model_state);
  }

  if (iree_status_is_ok(status)) {
    // Set up segment buffer at computed offset.
    uint8_t* base = (uint8_t*)state_storage.data;
    state->segments = (iree_tokenizer_segment_t*)(base + segments_offset);

    // Initialize special token match state (runs before normalizer).
    iree_tokenizer_special_tokens_encode_state_initialize(
        &state->special_token_match);

    // Initialize post-normalization special token match state (runs after
    // normalizer, before segmenter).
    iree_tokenizer_special_tokens_encode_state_initialize(
        &state->special_token_match_post);

    // Set up postprocessor state if ADD_SPECIAL_TOKENS is requested.
    // The initializer is a no-op when the template has no special tokens.
    if (iree_all_bits_set(flags,
                          IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS)) {
      iree_tokenizer_postprocessor_encode_state_initialize(
          &tokenizer->postprocessor, &tokenizer->postprocessor.single,
          &state->postprocessor);
    }

    state->pending_special_token = -1;
    state->first_consumed_by_special_token = false;
    state->in_finalize_mode = false;

    *out_state = state;
  } else {
    // Deinitialize any partially-initialized sub-states.
    iree_tokenizer_encode_state_deinitialize(state);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_tokenizer_encode_state_deinitialize(
    iree_tokenizer_encode_state_t* state) {
  if (!state) return;

  IREE_TRACE_ZONE_BEGIN(z0);
  // Deinitialize stage states (does not free storage).
  iree_tokenizer_model_state_deinitialize(state->model_state);
  iree_tokenizer_segmenter_state_deinitialize(state->segmenter_state);
  iree_tokenizer_normalizer_state_deinitialize(state->normalizer_state);

  memset(state, 0, sizeof(*state));
  IREE_TRACE_ZONE_END(z0);
}

void iree_tokenizer_encode_state_reset(iree_tokenizer_encode_state_t* state,
                                       iree_tokenizer_encode_flags_t flags) {
  if (!state) return;
  const iree_tokenizer_t* tokenizer = state->tokenizer;

  // Deinitialize component states (may have pending data to clear).
  iree_tokenizer_model_state_deinitialize(state->model_state);
  iree_tokenizer_segmenter_state_deinitialize(state->segmenter_state);
  iree_tokenizer_normalizer_state_deinitialize(state->normalizer_state);

  // Reset ring buffer positions.
  state->flags = flags;
  state->read_position = 0;
  state->write_position = 0;
  state->segmenter_view_start = 0;
  state->segment_count = 0;
  state->segments_consumed = 0;

  // Re-initialize component states (storage layout unchanged).
  if (state->normalizer_state) {
    iree_tokenizer_normalizer_state_initialize(tokenizer->normalizer,
                                               (void*)state->normalizer_state,
                                               &state->normalizer_state);
  }
  if (state->segmenter_state) {
    iree_tokenizer_segmenter_state_initialize(tokenizer->segmenter,
                                              (void*)state->segmenter_state,
                                              &state->segmenter_state);
  }
  if (state->model_state) {
    iree_tokenizer_model_state_initialize(
        tokenizer->model, (void*)state->model_state, &state->model_state);
  }

  // Reset special token match state.
  iree_tokenizer_special_tokens_encode_state_initialize(
      &state->special_token_match);

  // Reset post-normalization special token match state.
  iree_tokenizer_special_tokens_encode_state_initialize(
      &state->special_token_match_post);

  // Reset postprocessor state if ADD_SPECIAL_TOKENS is requested.
  if (iree_all_bits_set(flags, IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS)) {
    iree_tokenizer_postprocessor_encode_state_initialize(
        &tokenizer->postprocessor, &tokenizer->postprocessor.single,
        &state->postprocessor);
  } else {
    memset(&state->postprocessor, 0, sizeof(state->postprocessor));
  }

  state->pending_special_token = -1;
  state->first_consumed_by_special_token = false;
  state->in_finalize_mode = false;
}

// Returns true if the encode pipeline has content that must be emitted before
// a special token: ring buffer data, pending segments, or normalizer-buffered
// data (e.g., NFC combining sequences).
static inline bool iree_tokenizer_encode_state_pipeline_has_content(
    const iree_tokenizer_encode_state_t* state) {
  if (state->write_position > state->read_position) return true;
  if (state->segment_count > 0) return true;
  if (state->normalizer_state &&
      iree_tokenizer_normalizer_state_has_pending(state->normalizer_state)) {
    return true;
  }
  return false;
}

bool iree_tokenizer_encode_state_has_pending(
    const iree_tokenizer_encode_state_t* state) {
  IREE_ASSERT_ARGUMENT(state);

  // Check if any stage has pending data (ordered by pipeline position).
  // Deferred special token waiting for pipeline to flush.
  if (state->pending_special_token >= 0) {
    return true;
  }
  // Special token matching runs first, before normalizer.
  if (iree_tokenizer_special_tokens_encode_state_has_partial(
          &state->special_token_match)) {
    return true;
  }
  if (state->segment_count > state->segments_consumed) return true;
  if (state->write_position > state->segmenter_view_start) {
    return true;
  }
  if (state->normalizer_state &&
      iree_tokenizer_normalizer_state_has_pending(state->normalizer_state)) {
    return true;
  }
  // Post-normalization special token matching runs after normalizer output.
  if (iree_tokenizer_special_tokens_encode_state_has_partial(
          &state->special_token_match_post)) {
    return true;
  }
  if (state->segmenter_state &&
      iree_tokenizer_segmenter_state_has_pending(state->segmenter_state)) {
    return true;
  }
  if (state->model_state &&
      iree_tokenizer_model_state_has_pending(state->model_state)) {
    return true;
  }

  if (iree_tokenizer_postprocessor_encode_state_has_pending(
          &state->postprocessor)) {
    return true;
  }

  return false;
}

// Feeds ring buffer data [read_position, write_position) directly to the
// model as a single segment, bypassing the segmenter. Handles physical
// wrap-around via mirror copy. When |is_partial| is true, the model preserves
// state for continuation (more bytes will follow); when false, the model
// processes through to completion (FLUSH transition).
static iree_status_t iree_tokenizer_encode_ring_as_segment(
    iree_tokenizer_encode_state_t* state, bool is_partial,
    iree_tokenizer_token_output_t sub_output,
    iree_host_size_t* out_tokens_written, bool* out_segment_complete) {
  *out_tokens_written = 0;
  *out_segment_complete = false;

  // Guard against invariant violation: read_position must not exceed
  // write_position. Check BEFORE subtraction to avoid underflow.
  if (state->read_position >= state->write_position) {
    return iree_ok_status();
  }
  iree_host_size_t data_length = state->write_position - state->read_position;
  iree_host_size_t capacity = state->capacity_mask + 1;
  iree_host_size_t physical_read = state->read_position & state->capacity_mask;
  iree_host_size_t physical_write =
      state->write_position & state->capacity_mask;

  // Mirror copy if data wraps around the physical buffer end.
  if (physical_write <= physical_read && physical_write > 0) {
    memcpy(state->transform_buffer.data + capacity,
           state->transform_buffer.data, physical_write);
  }

  iree_tokenizer_segment_t segment = {
      .start = physical_read,
      .end = physical_read + data_length,
  };
  iree_tokenizer_segment_list_t segment_list = {
      .count = 1,
      .values = &segment,
      .last_is_partial = is_partial,
  };

  iree_host_size_t segments_consumed = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_model_state_encode(
      state->model_state,
      iree_make_const_byte_span(state->transform_buffer.data,
                                state->transform_buffer.data_length),
      segment_list, sub_output, &segments_consumed, out_tokens_written));
  *out_segment_complete = (segments_consumed > 0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// LSTRIP Whitespace Stripping Helpers
//===----------------------------------------------------------------------===//

// Returns the flags for a token given its ID by searching the special tokens
// collection. Returns NONE if not found (which shouldn't happen for valid
// matches).
static iree_tokenizer_special_token_flags_t
iree_tokenizer_special_tokens_get_flags_for_id(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_tokenizer_token_id_t token_id) {
  for (iree_host_size_t i = 0; i < special_tokens->count; ++i) {
    if (special_tokens->ids[i] == token_id) {
      return special_tokens->flags[i];
    }
  }
  return IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE;
}

// Strips trailing whitespace from a safe prefix length when a potential
// special token at position |safe| has the LSTRIP flag. This prevents
// the whitespace immediately preceding an LSTRIP token from being tokenized.
//
// |special_tokens|: the special tokens collection to check
// |input|: the input text being processed
// |safe|: the position where a special token could start (may be modified)
// |check_result|: the result from a speculative match at position |safe|
// |check_id|: the matched token ID (only valid if check_result is MATCHED)
// |check_state|: the match state (contains partial_token_index for NEED_MORE)
//
// Returns the adjusted safe prefix length with trailing whitespace removed
// if the potential token has LSTRIP.
static iree_host_size_t iree_tokenizer_strip_lstrip_whitespace(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_string_view_t input, iree_host_size_t safe,
    iree_tokenizer_special_tokens_match_result_t check_result,
    iree_tokenizer_token_id_t check_id,
    const iree_tokenizer_special_tokens_encode_state_t* check_state) {
  // Get flags for the potential token.
  iree_tokenizer_special_token_flags_t flags = 0;
  if (check_result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED) {
    flags = iree_tokenizer_special_tokens_get_flags_for_id(special_tokens,
                                                           check_id);
  } else if (check_result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE) {
    // For partial matches, check the token being matched.
    flags = special_tokens->flags[check_state->partial_token_index];
  }

  // If LSTRIP is set, strip trailing whitespace from the safe prefix.
  if (iree_any_bit_set(flags, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP)) {
    while (safe > 0 && (uint8_t)input.data[safe - 1] <= 0x20) {
      --safe;
    }
  }

  return safe;
}

//===----------------------------------------------------------------------===//
// Post-Normalization Special Token Matching
//===----------------------------------------------------------------------===//

// Result from post-normalization special token matching.
typedef enum iree_tokenizer_post_norm_match_result_e {
  // No special token could match anywhere in the view. Segmenter can consume
  // the entire normalized text.
  IREE_TOKENIZER_POST_NORM_NO_MATCH = 0,
  // A special token was matched and emitted. Caller should re-enter loop.
  IREE_TOKENIZER_POST_NORM_MATCHED,
  // Partial match in progress - waiting for more normalized output. Block
  // segmenter from consuming partial match bytes.
  IREE_TOKENIZER_POST_NORM_NEED_MORE,
  // A special token could start at |out_segment_limit| bytes into the view.
  // Segmenter should only consume bytes up to that limit, then re-check.
  IREE_TOKENIZER_POST_NORM_LIMIT,
} iree_tokenizer_post_norm_match_result_t;

// Matches post-normalization special tokens in the ring buffer.
// Called after normalizer writes output, before segmenter reads.
//
// Post-norm tokens have normalized=true in HuggingFace added_tokens.
// They are matched against NORMALIZED text, not raw input. For example, a token
// "yesterday" with a lowercasing normalizer would match input "Yesterday".
//
// The matching region is [segmenter_view_start, write_position) - the
// normalized bytes that haven't been processed by the segmenter yet.
//
// Returns:
//   NO_MATCH: No special token in view. Segmenter can process all bytes.
//   MATCHED: Token emitted. Re-enter loop.
//   NEED_MORE: Partial match. Block segmenter, wait for input.
//   LIMIT: Potential match at *out_segment_limit. Segmenter should only
//          process that many bytes, then re-check.
static iree_tokenizer_post_norm_match_result_t
iree_tokenizer_encode_state_match_post_norm_special_tokens(
    iree_tokenizer_encode_state_t* state, const iree_tokenizer_t* tokenizer,
    iree_tokenizer_token_output_t output, iree_host_size_t* total_tokens,
    iree_host_size_t* out_segment_limit) {
  *out_segment_limit = IREE_HOST_SIZE_MAX;

  // Skip if no post-norm tokens configured.
  if (iree_tokenizer_special_tokens_is_empty(
          &tokenizer->special_tokens_post_norm)) {
    return IREE_TOKENIZER_POST_NORM_NO_MATCH;
  }
  // Skip if no unprocessed normalized output available.
  if (state->write_position <= state->segmenter_view_start) {
    return IREE_TOKENIZER_POST_NORM_NO_MATCH;
  }

  // Get the normalized text view (segmenter_view_start to write_position).
  iree_string_view_t normalized_text =
      iree_tokenizer_ring_get_segment_view(state);
  if (normalized_text.size == 0) {
    return IREE_TOKENIZER_POST_NORM_NO_MATCH;
  }

  // When continuing a partial match (NEED_MORE), the match API contract
  // requires that we pass only new bytes, not bytes already consumed. The
  // match_position tracks how many bytes have been matched so far.
  iree_host_size_t already_matched =
      state->special_token_match_post.match_position;
  if (already_matched > 0) {
    // Slice to only new bytes.
    if (already_matched >= normalized_text.size) {
      // No new bytes available yet - keep waiting.
      return IREE_TOKENIZER_POST_NORM_NEED_MORE;
    }
    normalized_text.data += already_matched;
    normalized_text.size -= already_matched;
  }

  // O(1) rejection: check if any special token could start at this position.
  // If safe_prefix_length > 0, no token starts in the first `safe` bytes.
  iree_host_size_t safe = 0;
  if (already_matched == 0) {
    safe = iree_tokenizer_special_tokens_safe_prefix_length(
        &tokenizer->special_tokens_post_norm, normalized_text);
    if (safe == normalized_text.size) {
      return IREE_TOKENIZER_POST_NORM_NO_MATCH;  // No possible match.
    }
    // If safe > 0, a special token could start at position `safe`. Check if it
    // actually matches/could match before reporting LIMIT. This avoids false
    // LIMITs for characters like `<` in `<=` that look like they could start a
    // token like `<|EOT|>` but don't.
    if (safe > 0) {
      iree_string_view_t check_view = {
          .data = normalized_text.data + safe,
          .size = normalized_text.size - safe,
      };
      iree_host_size_t check_length = 0;
      iree_tokenizer_token_id_t check_id = 0;
      iree_tokenizer_special_tokens_encode_state_t check_state = {0};
      iree_tokenizer_special_tokens_match_result_t check_result =
          iree_tokenizer_special_tokens_match(
              &tokenizer->special_tokens_post_norm, check_view, &check_length,
              &check_id, &check_state);
      if (check_result == IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH) {
        // Not a special token (e.g., `<=` is not `<|EOT|>`). Let segmenter see
        // all bytes including the `<`.
        return IREE_TOKENIZER_POST_NORM_NO_MATCH;
      }
      // In finalize mode, NEED_MORE means the partial prefix will never become
      // a complete special token (no more input is coming). Treat it as
      // NO_MATCH so the segmenter can process all bytes including the potential
      // prefix. This prevents " <" at end-of-input from being split into " " +
      // "<".
      if (check_result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE &&
          state->in_finalize_mode) {
        return IREE_TOKENIZER_POST_NORM_NO_MATCH;
      }
      // A special token matches or could match at position `safe`. Strip
      // trailing whitespace if the token has LSTRIP - this prevents the
      // whitespace from being tokenized when it should be consumed by the
      // special token match (RoBERTa/BART <mask> with lstrip=true).
      iree_host_size_t original_safe = safe;
      safe = iree_tokenizer_strip_lstrip_whitespace(
          &tokenizer->special_tokens_post_norm, normalized_text, safe,
          check_result, check_id, &check_state);
      if (safe == 0 && original_safe > 0) {
        // All prefix was whitespace. Skip those bytes and let the match
        // proceed on the next call. Update prev_byte context so LSTRIP check
        // succeeds when using the real state.
        uint8_t last_whitespace =
            (uint8_t)normalized_text.data[original_safe - 1];
        state->special_token_match_post.prev_byte_plus_one =
            last_whitespace + 1;
        state->special_token_match_post.at_start_of_input = false;
        state->segmenter_view_start += original_safe;
        // Advance read_position if no segments are pending.
        if (state->segment_count == state->segments_consumed) {
          state->read_position = state->segmenter_view_start;
        }
      }
      // Return LIMIT so the segmenter processes the prefix bytes first. On
      // the next call, `safe` will be 0 and we'll match/wait properly.
      *out_segment_limit = safe;
      return IREE_TOKENIZER_POST_NORM_LIMIT;
    }
  }

  // Attempt to match (handles both fresh and continued partial matches).
  iree_host_size_t match_length = 0;
  iree_tokenizer_token_id_t match_id = 0;
  iree_tokenizer_special_tokens_match_result_t result =
      iree_tokenizer_special_tokens_match(
          &tokenizer->special_tokens_post_norm, normalized_text, &match_length,
          &match_id, &state->special_token_match_post);

  if (result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED) {
    // Emit the special token.
    if (*total_tokens < output.capacity) {
      output.token_ids[*total_tokens] = match_id;
      iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor,
                                                   output, *total_tokens, 1);
      // Post-norm special tokens don't map directly to raw input bytes (they
      // matched normalized text), so use zero-length offsets like the
      // postprocessor does for added special tokens.
      if (output.token_offsets) {
        output.token_offsets[*total_tokens].start = 0;
        output.token_offsets[*total_tokens].end = 0;
      }
      (*total_tokens)++;
    }
    // Advance segmenter_view_start past the matched bytes - the segmenter
    // won't see them. Include any bytes from a partial match continuation.
    iree_host_size_t skipped_bytes = already_matched + match_length;
    state->segmenter_view_start += skipped_bytes;
    // Advance read_position if no segments are pending. Pending segments
    // reference physical offsets starting at read_position - advancing it
    // prematurely would "free" memory the model still needs, corrupting data
    // if the ring wraps.
    if (state->segment_count == state->segments_consumed) {
      state->read_position = state->segmenter_view_start;
    }
    return IREE_TOKENIZER_POST_NORM_MATCHED;
  }

  if (result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE) {
    // Partial match in progress.
    if (state->in_finalize_mode) {
      // In finalize mode, NEED_MORE means this will never become a complete
      // special token. Clear any partial state and return NO_MATCH so the
      // segmenter processes these bytes normally.
      if (state->special_token_match_post.match_position > 0) {
        iree_tokenizer_special_tokens_encode_state_clear_partial(
            &state->special_token_match_post);
      }
      return IREE_TOKENIZER_POST_NORM_NO_MATCH;
    }
    // Wait for more normalized output. Don't advance any positions - the bytes
    // are still in the ring and will be re-examined once more input arrives.
    return IREE_TOKENIZER_POST_NORM_NEED_MORE;
  }

  // NO_MATCH: Let the segmenter process these bytes normally.
  // If there was a partial match, clear it - those bytes aren't a special
  // token.
  if (state->special_token_match_post.match_position > 0) {
    iree_tokenizer_special_tokens_encode_state_clear_partial(
        &state->special_token_match_post);
  }
  return IREE_TOKENIZER_POST_NORM_NO_MATCH;
}

// Attempts to emit a deferred special token.
// Returns true if a token was emitted (caller should re-pump).
static bool iree_tokenizer_try_emit_pending_special_token(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* total_tokens) {
  if (state->pending_special_token < 0) return false;

  // Only emit when pipeline is flushed: no ring buffer content, no segments,
  // and no normalizer-buffered data (e.g., NFC combining sequences).
  bool pipeline_flushed =
      !iree_tokenizer_encode_state_pipeline_has_content(state);
  if (!pipeline_flushed) return false;

  // Emit the deferred token.
  if (output.capacity > *total_tokens) {
    output.token_ids[*total_tokens] = state->pending_special_token;
    iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor, output,
                                                 *total_tokens, 1);
    (*total_tokens)++;
    state->pending_special_token = -1;
    return true;
  }
  return false;
}

// Strips trailing whitespace-only segments from the segment output.
// When the segmenter finalizes due to a LIMIT constraint (e.g., a special token
// could start at position N), trailing whitespace should stay with the next
// segment rather than becoming a standalone segment. For example, " n " before
// "<=" should produce " n" with the trailing space joining "<=".
//
// Modifies |segments_produced| and |bytes_consumed| to exclude any trailing
// whitespace-only segments.
static void iree_tokenizer_strip_trailing_whitespace_segments(
    iree_string_view_t normalized_text,
    iree_tokenizer_segment_output_t segment_output,
    iree_host_size_t* segments_produced, iree_host_size_t* bytes_consumed) {
  while (*segments_produced > 0) {
    iree_tokenizer_segment_t* last_segment =
        &segment_output.values[*segments_produced - 1];
    bool all_whitespace = true;
    for (iree_host_size_t i = last_segment->start; i < last_segment->end; i++) {
      if (normalized_text.data[i] != ' ' && normalized_text.data[i] != '\t' &&
          normalized_text.data[i] != '\n' && normalized_text.data[i] != '\r') {
        all_whitespace = false;
        break;
      }
    }
    if (!all_whitespace) break;
    // Remove this whitespace-only segment and don't consume those bytes.
    *bytes_consumed = last_segment->start;
    (*segments_produced)--;
  }
}

// Internal helper: run one iteration of the encode pipeline.
// Sets |out_made_progress| to true if data moved through the pipeline.
static iree_status_t iree_tokenizer_encode_state_pump(
    iree_tokenizer_encode_state_t* state, iree_string_view_t* chunk,
    iree_tokenizer_token_output_t* output, iree_host_size_t* total_tokens,
    bool* out_made_progress) {
  const iree_tokenizer_t* tokenizer = state->tokenizer;
  *out_made_progress = false;

  // Reclaim ring bytes consumed by the segmenter when no segments are pending.
  // The main reclamation path (below) only runs when segment_count >
  // segments_consumed, but when both are 0 the segmenter may have consumed
  // bytes (advancing segmenter_view_start) without producing segments. Those
  // bytes are dead space that prevents ring position reset, which can leave
  // the write position near the physical wrap boundary with insufficient
  // contiguous space for the normalizer to write multi-byte codepoints.
  if (state->segment_count == state->segments_consumed &&
      state->segmenter_view_start > state->read_position) {
    state->read_position = state->segmenter_view_start;
    // Reset ring positions to 0 when empty for full contiguous write space.
    if (state->read_position == state->write_position &&
        state->segmenter_view_start == state->write_position &&
        !(state->segmenter_state &&
          iree_tokenizer_segmenter_state_has_pending(state->segmenter_state))) {
      state->read_position = 0;
      state->segmenter_view_start = 0;
      state->write_position = 0;
    }
  }

  // Tokenize pending segments. Segments are produced by the segmenter and
  // queued for the model. Process them first to make room in the segment
  // buffer.
  if (state->segment_count > state->segments_consumed &&
      output->capacity > *total_tokens) {
    // Convert segment positions from logical to physical for model access.
    // Segments are guaranteed not to span the wrap boundary (contiguity).
    iree_host_size_t pending_count =
        state->segment_count - state->segments_consumed;
    iree_tokenizer_segment_t* pending_base =
        &state->segments[state->segments_consumed];
    for (iree_host_size_t i = 0; i < pending_count; ++i) {
      iree_host_size_t length = pending_base[i].end - pending_base[i].start;
      pending_base[i].start = pending_base[i].start & state->capacity_mask;
      pending_base[i].end = pending_base[i].start + length;
    }

    iree_tokenizer_segment_list_t pending_segments = {
        .count = pending_count,
        .values = pending_base,
    };
    iree_tokenizer_token_output_t sub_output = {
        .capacity = output->capacity - *total_tokens,
        .token_ids = &output->token_ids[*total_tokens],
        .token_offsets = output->token_offsets
                             ? &output->token_offsets[*total_tokens]
                             : NULL,
        .type_ids = output->type_ids ? &output->type_ids[*total_tokens] : NULL,
    };

    iree_host_size_t segments_consumed = 0;
    iree_host_size_t tokens_written = 0;
    // Pass full physical allocation - model accesses physical positions,
    // which may extend into the mirror region [capacity, 2*capacity) for
    // segments that span the logical wrap boundary.
    IREE_RETURN_IF_ERROR(iree_tokenizer_model_state_encode(
        state->model_state,
        iree_make_const_byte_span(state->transform_buffer.data,
                                  state->transform_buffer.data_length),
        pending_segments, sub_output, &segments_consumed, &tokens_written));

    // Assign type_ids to model-produced tokens based on sequence phase.
    iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor,
                                                 sub_output, 0, tokens_written);
    // Trim offsets on model tokens if requested (ByteLevel/RoBERTa behavior).
    iree_tokenizer_postprocessor_trim_token_offsets(
        &state->postprocessor, state->tokenizer->vocab, sub_output, 0,
        tokens_written);

    state->segments_consumed += segments_consumed;
    *total_tokens += tokens_written;
    if (segments_consumed > 0 || tokens_written > 0) {
      *out_made_progress = true;
    }

    // If all segments consumed, advance read pointer to reclaim space.
    // No data movement - ring buffer just advances the read position.
    if (state->segments_consumed >= state->segment_count) {
      state->segment_count = 0;
      state->segments_consumed = 0;
      // Reclaim bytes up to segmenter_view_start. This includes both bytes
      // that were converted to segments and bytes consumed by the segmenter
      // without producing segments (buffering for word boundaries, etc.).
      state->read_position = state->segmenter_view_start;

      // If the ring is now empty AND the segmenter has no pending state,
      // reset all positions to 0. This ensures we always have full
      // contiguous write space available, avoiding deadlock when write
      // position approaches the physical boundary with less than 4 bytes
      // remaining (not enough for a UTF-8 codepoint).
      //
      // The segmenter check is critical: when the segmenter holds a pending
      // regex match, its finalize will produce segments relative to
      // bytes_processed. The adjustment (+= segmenter_view_start) must cancel
      // the chunk_base subtraction, which requires segmenter_view_start to
      // equal bytes_processed (not 0).
      if (state->read_position == state->write_position &&
          state->segmenter_view_start == state->write_position &&
          !(state->segmenter_state &&
            iree_tokenizer_segmenter_state_has_pending(
                state->segmenter_state))) {
        state->read_position = 0;
        state->segmenter_view_start = 0;
        state->write_position = 0;
      }
    }
  }

  // Emit deferred pre-norm special token if pipeline has flushed.
  if (iree_tokenizer_try_emit_pending_special_token(state, *output,
                                                    total_tokens)) {
    *out_made_progress = true;
    return iree_ok_status();
  }

  // Force-flush normalizer when a special token is pending but the normalizer
  // holds data that blocks pipeline drain. The special token boundary acts as
  // a segment end for the normalizer's combining sequence.
  //
  // We use state_finalize() rather than state_process() with SEGMENT_END
  // because Sequence normalizers don't process the SEGMENT_END flag when given
  // empty input (the processing loop requires input.size > 0 to run).
  if (state->pending_special_token >= 0 && state->normalizer_state &&
      iree_tokenizer_normalizer_state_has_pending(state->normalizer_state)) {
    iree_host_size_t write_space = iree_tokenizer_ring_writable_space(state);
    if (write_space > 0) {
      iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
          state->write_position, state->capacity_mask);
      iree_mutable_string_view_t norm_output = iree_make_mutable_string_view(
          (char*)state->transform_buffer.data + physical_write, write_space);
      iree_host_size_t bytes_written = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_finalize(
          state->normalizer_state, norm_output, &bytes_written));
      iree_tokenizer_ring_fixup_write_wrap(state, physical_write,
                                           bytes_written);
      state->write_position += bytes_written;
      if (bytes_written > 0) {
        *out_made_progress = true;
        return iree_ok_status();
      }
    }
  }

  // Match post-normalization special tokens before segmenter runs.
  // These tokens have normalized=true and are matched after the normalizer
  // transforms the input. If matched, the token is emitted directly and we
  // re-enter the loop to check for more matches or continue processing.
  //
  // IMPORTANT: Skip post-norm matching when the segmenter has pending state.
  // When the segmenter returns consumed=0 with pending internal state (e.g.,
  // sequence segmenter mid-expansion), it stores absolute byte offsets into
  // the input view. Post-norm matching can advance segmenter_view_start
  // (LSTRIP whitespace skip or matched token skip), which shrinks the ring
  // view. The segmenter's stored offsets then exceed the new view size,
  // causing unsigned underflow when computing remaining bytes. Let the
  // segmenter finish its pending work first; post-norm matching will run on
  // the next pump iteration after the segmenter commits its consumed bytes.
  iree_host_size_t post_norm_segment_limit = IREE_HOST_SIZE_MAX;
  if (!(state->segmenter_state &&
        iree_tokenizer_segmenter_state_has_pending(state->segmenter_state))) {
    iree_tokenizer_post_norm_match_result_t post_norm_result =
        iree_tokenizer_encode_state_match_post_norm_special_tokens(
            state, tokenizer, *output, total_tokens, &post_norm_segment_limit);
    switch (post_norm_result) {
      case IREE_TOKENIZER_POST_NORM_MATCHED:
        // Token emitted. Re-enter loop to check for more matches.
        *out_made_progress = true;
        return iree_ok_status();
      case IREE_TOKENIZER_POST_NORM_NEED_MORE:
        // Partial match - waiting for more input. Block segmenter by setting
        // limit to 0, but continue to normalize step so we can accumulate more
        // bytes in the ring buffer. The next pump iteration will retry the
        // match with more data.
        post_norm_segment_limit = 0;
        break;
      case IREE_TOKENIZER_POST_NORM_LIMIT:
        // Special token could start at post_norm_segment_limit. Let segmenter
        // process only that many bytes (handled below).
        break;
      case IREE_TOKENIZER_POST_NORM_NO_MATCH:
        // No special token. Segmenter can process all bytes.
        break;
    }
  }

  // Segment normalized text. Requires: segment buffer has room, ring has data.
  // get_segment_view handles wrap-around by copying to mirror region if needed.
  // Truncate the view to exclude any incomplete UTF-8 trailing sequence. The
  // ring may contain partial codepoints when input chunks split mid-sequence
  // (passthrough path) - the segmenter must only see complete codepoints.
  //
  // When partial mode is active, the segmenter is bypassed: those bytes are
  // being fed directly to the model. Continuing to feed the segmenter would
  // cause it to produce segments for bytes already processed, with positions
  // referencing reclaimed ring buffer space.
  iree_string_view_t normalized_text =
      iree_tokenizer_ring_get_segment_view(state);
  if (normalized_text.size > 0) {
    normalized_text.size -= iree_unicode_utf8_incomplete_tail_length(
        normalized_text.data, normalized_text.size);
  }
  // Apply post-norm segment limit: if a special token could start at position
  // N, only let the segmenter see bytes [0, N). After processing, post-norm
  // matching will get another chance to match at the new segmenter_view_start.
  if (post_norm_segment_limit < normalized_text.size) {
    normalized_text.size = post_norm_segment_limit;
  }
  if (!state->has_partial_segment &&
      state->segment_count < tokenizer->segment_batch_size &&
      normalized_text.size > 0) {
    iree_tokenizer_segment_output_t segment_output = {
        .capacity = tokenizer->segment_batch_size - state->segment_count,
        .values = &state->segments[state->segment_count],
    };

    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t segments_produced = 0;
    if (state->segmenter_state) {
      IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_process(
          state->segmenter_state, normalized_text, segment_output,
          &bytes_consumed, &segments_produced));
    } else {
      // NULL segmenter: passthrough mode - emit entire input as one segment.
      segment_output.values[0].start = 0;
      segment_output.values[0].end = normalized_text.size;
      bytes_consumed = normalized_text.size;
      segments_produced = 1;
    }

    // Handle post-norm LIMIT deadlock: when we limited the segmenter view due
    // to a potential special token, but the segmenter couldn't make progress
    // (the limited bytes don't form a complete segment), we force the segmenter
    // to finalize on the limited view. This produces proper segment splits
    // while keeping the special token position intact for matching.
    //
    // Example: for "hello <|special|>", after segmenting "hello", the remaining
    // " <|special|>" has safe_prefix=1 (the space). The space alone doesn't
    // form a complete segment boundary, so we finalize the segmenter to force
    // it to produce segments up to the limit.
    //
    // Note: We must NOT create a synthetic segment for the entire prefix, as
    // that would bypass proper segmentation. For example, "    if<" should
    // produce segments [0,3) and [3,6) for "   " and " if", not a single [0,6).
    if (bytes_consumed == 0 && segments_produced == 0 &&
        post_norm_segment_limit < IREE_HOST_SIZE_MAX &&
        post_norm_segment_limit > 0 && segment_output.capacity > 0 &&
        state->segmenter_state) {
      // Finalize the segmenter on the limited view to get proper splits.
      IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_finalize(
          state->segmenter_state, normalized_text, segment_output,
          &segments_produced));
      bytes_consumed = normalized_text.size;

      // When finalizing due to LIMIT for regular text (not special tokens),
      // trailing whitespace-only segments should not be consumed - they belong
      // with the next segment. However, for post-norm special tokens,
      // whitespace before the token SHOULD be consumed because the special
      // token boundary itself acts as a word boundary.
      //
      // During pump() (non-finalize mode), we don't yet know if the thing after
      // LIMIT is a real special token or a false positive (like `<` that isn't
      // `<|endoftext|>`). Always strip whitespace during pump to allow retry
      // with more context. In finalize mode, the post-norm check returns
      // NO_MATCH instead of LIMIT (lines 1554-1558), so the segmenter sees all
      // bytes and produces correct segments - this path won't be reached.
      //
      // The infinite loop concern (stripping whitespace forever) is handled by
      // finalize mode: when finalize() is called, in_finalize_mode=true causes
      // the LIMIT to become NO_MATCH, breaking the loop.
      bool has_confirmed_special_token_waiting =
          (bytes_consumed == post_norm_segment_limit &&
           !iree_tokenizer_special_tokens_is_empty(
               &tokenizer->special_tokens_post_norm) &&
           state->in_finalize_mode);
      if (!has_confirmed_special_token_waiting) {
        iree_tokenizer_strip_trailing_whitespace_segments(
            normalized_text, segment_output, &segments_produced,
            &bytes_consumed);
      }
    }

    // Handle pending special token deadlock: when a special token is waiting
    // to be emitted but the segmenter didn't produce segments (waiting for
    // word boundary), create a synthetic segment for all ring content. The
    // special token itself acts as a word boundary.
    if (bytes_consumed == 0 && segments_produced == 0 &&
        state->pending_special_token >= 0 && normalized_text.size > 0 &&
        segment_output.capacity > 0) {
      segment_output.values[0].start = 0;
      segment_output.values[0].end = normalized_text.size;
      bytes_consumed = normalized_text.size;
      segments_produced = 1;
    }

    // Capture segmenter_view_start before mutation for offset adjustment.
    // Segments use this captured value to convert from relative to absolute
    // offsets. This makes segments immune to later state mutations (e.g.,
    // post-norm special token matching advancing segmenter_view_start).
    iree_host_size_t view_start = state->segmenter_view_start;

    // Convert segment offsets from relative to absolute (logical cumulative).
    // These will be converted to physical when passed to model.
    for (iree_host_size_t i = 0; i < segments_produced; ++i) {
      state->segments[state->segment_count + i].start += view_start;
      state->segments[state->segment_count + i].end += view_start;
    }

    // Advance segmenter_view_start for next call.
    state->segmenter_view_start += bytes_consumed;
    state->segment_count += segments_produced;
    if (bytes_consumed > 0 || segments_produced > 0) {
      *out_made_progress = true;
    }
  }

  // Match special tokens in raw input (runs BEFORE normalizer).
  // Special tokens like <|endoftext|> must be matched literally before the
  // normalizer or pre-tokenizer could transform them.
  //
  // The match() function handles continuation: if state->match_position > 0,
  // we're continuing a partial match and |chunk| contains only NEW bytes.
  // On NEED_MORE, we consume the bytes and track progress in state. The token
  // content itself serves as the "buffer" for reconstruction via get_partial().
  iree_host_size_t normalize_limit = IREE_HOST_SIZE_MAX;
  if (!iree_tokenizer_special_tokens_is_empty(&tokenizer->special_tokens) &&
      chunk->size > 0) {
    // Continuing a partial match? Pass chunk directly to match().
    // Starting fresh? Check if first byte could start a special token.
    bool should_match = state->special_token_match.match_position > 0;
    if (!should_match) {
      iree_host_size_t safe = iree_tokenizer_special_tokens_safe_prefix_length(
          &tokenizer->special_tokens,
          iree_make_string_view(chunk->data, chunk->size));
      if (safe == 0) {
        should_match = true;
      } else {
        // Check if the potential token at position safe has LSTRIP.
        // If so, strip trailing whitespace from the normalize limit to prevent
        // the whitespace from being tokenized (RoBERTa/BART <mask> with
        // lstrip=true).
        if (safe < chunk->size) {
          iree_string_view_t check_view = {
              .data = chunk->data + safe,
              .size = chunk->size - safe,
          };
          iree_host_size_t check_length = 0;
          iree_tokenizer_token_id_t check_id = 0;
          iree_tokenizer_special_tokens_encode_state_t check_state = {0};
          iree_tokenizer_special_tokens_match_result_t check_result =
              iree_tokenizer_special_tokens_match(&tokenizer->special_tokens,
                                                  check_view, &check_length,
                                                  &check_id, &check_state);
          if (check_result != IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH) {
            iree_host_size_t original_safe = safe;
            safe = iree_tokenizer_strip_lstrip_whitespace(
                &tokenizer->special_tokens,
                iree_make_string_view(chunk->data, chunk->size), safe,
                check_result, check_id, &check_state);
            if (safe == 0 && original_safe > 0) {
              // All prefix was whitespace. Skip those bytes and match the
              // special token directly. The whitespace is consumed by LSTRIP.
              // Update prev_byte context to be the last whitespace byte so
              // the LSTRIP match succeeds when using the real state.
              uint8_t last_whitespace = (uint8_t)chunk->data[original_safe - 1];
              state->special_token_match.prev_byte_plus_one =
                  last_whitespace + 1;
              state->special_token_match.at_start_of_input = false;
              chunk->data += original_safe;
              chunk->size -= original_safe;
              *out_made_progress = true;
              should_match = true;
            }
            // If safe > 0 (partial whitespace stripped), we just reduce the
            // normalize limit to exclude trailing whitespace before the token.
          }
        }
        if (!should_match) {
          normalize_limit = safe;
        }
      }
    }

    if (should_match) {
      iree_host_size_t match_length = 0;
      iree_tokenizer_token_id_t match_id = 0;
      iree_tokenizer_special_tokens_match_result_t result =
          iree_tokenizer_special_tokens_match(
              &tokenizer->special_tokens,
              iree_make_string_view(chunk->data, chunk->size), &match_length,
              &match_id, &state->special_token_match);

      if (result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED) {
        // Track if this special token matched at position 0 of the original
        // input. This affects prepend_scheme="first" semantics: text following
        // a position-0 special token should NOT get metaspace prepended.
        if (state->special_token_match.at_start_of_input) {
          state->first_consumed_by_special_token = true;
        }

        // Consume matched bytes from input.
        chunk->data += match_length;
        chunk->size -= match_length;
        *out_made_progress = true;

        // Check if pipeline has content that must be emitted first.
        bool pipeline_has_content =
            iree_tokenizer_encode_state_pipeline_has_content(state);
        if (pipeline_has_content) {
          // Defer emission until pipeline flushes.
          state->pending_special_token = match_id;
        } else {
          // Pipeline empty, emit immediately.
          if (output->capacity > *total_tokens) {
            output->token_ids[*total_tokens] = match_id;
            iree_tokenizer_postprocessor_assign_type_ids(
                &state->postprocessor, *output, *total_tokens, 1);
            (*total_tokens)++;
          }
        }
        return iree_ok_status();
      } else if (result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE) {
        // Partial match. Consume bytes (tracked in state), wait for more.
        chunk->data += chunk->size;
        chunk->size = 0;
        *out_made_progress = true;
        return iree_ok_status();
      }
      // NO_MATCH. If we had a partial match in progress, recover via finalize
      // or drain the partial bytes through the normalizer.
      if (state->special_token_match.match_position > 0) {
        // Drain partial match bytes through normalizer (one at a time).
        // get_partial() reconstructs the bytes from the token data itself.
        normalize_limit = 0;
      } else {
        // Match at position 0 definitively failed (not NEED_MORE). Find where
        // the next potential special token could start and normalize up to it.
        // This allows normalizing e.g. "<=" fully when we know it's not
        // "<|...".
        //
        // Without this, we'd normalize only 1 byte (the '<'), causing post-norm
        // matching to see just '<' and return NEED_MORE (could be start of
        // special token), which incorrectly splits tokens like " <=" into " "
        // and "<=".
        if (chunk->size > 1) {
          iree_host_size_t next_safe =
              iree_tokenizer_special_tokens_safe_prefix_length(
                  &tokenizer->special_tokens,
                  iree_make_string_view(chunk->data + 1, chunk->size - 1));
          normalize_limit = 1 + next_safe;
        } else {
          normalize_limit = 1;
        }
      }
    }
  }

  // Drain partial special token match after NO_MATCH.
  // The bytes are reconstructed from the token data via get_partial().
  // Uses drain_position to track which byte to emit (not match_position).
  if (state->special_token_match.match_position > 0 && normalize_limit == 0) {
    iree_host_size_t write_space = iree_tokenizer_ring_writable_space(state);
    if (write_space > 0) {
      // Get the partial match bytes from the token data.
      uint8_t partial_buffer[256];  // Max special token length.
      iree_host_size_t partial_size =
          iree_tokenizer_special_tokens_encode_state_get_partial(
              &state->special_token_match, &tokenizer->special_tokens,
              partial_buffer);
      if (state->special_token_match.drain_position < partial_size) {
        // Feed next byte to normalizer (at drain_position, not always 0).
        iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
            state->write_position, state->capacity_mask);
        iree_host_size_t bytes_written = 0;
        iree_host_size_t drain_offset =
            state->special_token_match.drain_position;
        iree_string_view_t drain_byte = iree_make_string_view(
            (const char*)partial_buffer + drain_offset, 1);
        if (state->normalizer_state) {
          iree_host_size_t bytes_consumed = 0;
          iree_mutable_string_view_t norm_output = {
              .data = (char*)state->transform_buffer.data + physical_write,
              .size = write_space,
          };
          // Draining partial buffer: no segment boundary (FLAG_NONE).
          IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
              state->normalizer_state, drain_byte, norm_output,
              IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &bytes_consumed,
              &bytes_written));
        } else {
          memcpy((char*)state->transform_buffer.data + physical_write,
                 partial_buffer + drain_offset, 1);
          bytes_written = 1;
        }
        iree_tokenizer_ring_fixup_write_wrap(state, physical_write,
                                             bytes_written);
        state->write_position += bytes_written;
        // Increment drain_position to track draining progress.
        state->special_token_match.drain_position++;
        // Clear state when all bytes drained.
        if (state->special_token_match.drain_position ==
            state->special_token_match.match_position) {
          iree_tokenizer_special_tokens_encode_state_clear_partial(
              &state->special_token_match);
        }
        *out_made_progress = true;
        return iree_ok_status();
      }
    }
  }

  // Normalize input and write to ring buffer. Requires: ring has write space.
  // If no normalizer is configured, copy input directly (passthrough).
  //
  // Skip normalization when there's a pending special token waiting to emit.
  // This forces the pump to drain existing ring content first, ensuring correct
  // output order: [content before special token, special token, content after].
  iree_host_size_t write_space = iree_tokenizer_ring_writable_space(state);
  if (write_space > 0 && chunk->size > 0 && normalize_limit > 0 &&
      state->pending_special_token < 0) {
    iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
        state->write_position, state->capacity_mask);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t bytes_written = 0;

    if (state->normalizer_state) {
      // Normalizer requires complete UTF-8 codepoints (see normalizer.h
      // contract). Truncate the view to exclude any incomplete trailing
      // sequence so the normalizer always receives valid boundaries.
      // Also respect normalize_limit from special token matching.
      iree_host_size_t max_consume =
          iree_min(iree_min(chunk->size, write_space), normalize_limit);
      iree_host_size_t incomplete_tail =
          iree_unicode_utf8_incomplete_tail_length(chunk->data, max_consume);
      max_consume -= incomplete_tail;
      if (max_consume == 0) {
        // Entire consumable region is an incomplete sequence. No progress
        // until more input arrives to complete the codepoint.
        return iree_ok_status();
      }
      iree_string_view_t safe_chunk =
          iree_make_string_view(chunk->data, max_consume);
      iree_mutable_string_view_t norm_output = {
          .data = (char*)state->transform_buffer.data + physical_write,
          .size = write_space,
      };
      // Signal segment end when normalize_limit restricts us to less than the
      // full input (indicates special token boundary ahead). This prevents
      // deadlocks in normalizers that use lazy consumption (e.g., Strip
      // normalizer waiting for lookahead that will never arrive because the
      // tokenizer won't feed past the special token boundary).
      iree_tokenizer_normalizer_flags_t flags =
          (normalize_limit < chunk->size)
              ? IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END
              : IREE_TOKENIZER_NORMALIZER_FLAG_NONE;

      // Signal to prepend normalizer that position 0 was consumed by a special
      // token, so prepend_scheme="first" should skip prepending for this text.
      if (state->first_consumed_by_special_token) {
        flags |= IREE_TOKENIZER_NORMALIZER_FLAG_FIRST_CONSUMED;
        // Clear after passing: only the first normalization gets this flag.
        state->first_consumed_by_special_token = false;
      }

      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
          state->normalizer_state, safe_chunk, norm_output, flags,
          &bytes_consumed, &bytes_written));

      // Handle normalizer stuck on potential trailing content. If no bytes were
      // consumed and this is the only remaining input (normalize_limit covers
      // all input, meaning no special token boundary), the normalizer is
      // waiting for lookahead that won't arrive. Retry with SEGMENT_END to
      // force it to treat the content as definitely trailing.
      if (bytes_consumed == 0 && bytes_written == 0 && max_consume > 0 &&
          normalize_limit >= chunk->size &&
          (flags & IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END) == 0) {
        IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
            state->normalizer_state, safe_chunk, norm_output,
            flags | IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END, &bytes_consumed,
            &bytes_written));
      }
    } else {
      // Passthrough: copy raw bytes directly to the ring buffer. Partial
      // UTF-8 sequences at the end are fine here - the ring accumulates bytes
      // across feed() calls, and Step 2 truncates the segmenter view to
      // exclude any incomplete trailing codepoint.
      // Also respect normalize_limit from special token matching.
      bytes_consumed =
          iree_min(iree_min(chunk->size, write_space), normalize_limit);
      memcpy((char*)state->transform_buffer.data + physical_write, chunk->data,
             bytes_consumed);
      bytes_written = bytes_consumed;
    }

    iree_tokenizer_ring_fixup_write_wrap(state, physical_write, bytes_written);
    chunk->data += bytes_consumed;
    chunk->size -= bytes_consumed;
    state->write_position += bytes_written;
    if (bytes_consumed > 0 || bytes_written > 0) {
      *out_made_progress = true;
    }
  }

  // Handle ring full with no segment boundaries. When no progress was made,
  // has data but no segment boundaries (split=false), enter partial segment
  // mode. The model processes ring data directly, emitting frozen tokens and
  // reclaiming bytes to free ring space for more normalization.
  if (!*out_made_progress && chunk->size > 0 &&
      state->write_position > state->read_position &&
      state->segment_count <= state->segments_consumed) {
    state->has_partial_segment = true;
  }
  if (state->has_partial_segment &&
      state->write_position > state->read_position &&
      output->capacity > *total_tokens) {
    // A pending special token is a segment boundary: the ring content before
    // it must be fully tokenized so the ring can drain and the special token
    // can emit. Without this, the model's holdback zone (which reserves the
    // last max_token_length bytes for future context) prevents progress when
    // the ring has fewer bytes than max_token_length, creating a three-way
    // deadlock: model won't process (holdback), normalizer won't feed
    // (pending special token blocks it), special token won't emit (ring not
    // empty).
    bool force_complete = state->pending_special_token >= 0;
    iree_tokenizer_token_output_t sub_output = {
        .capacity = output->capacity - *total_tokens,
        .token_ids = &output->token_ids[*total_tokens],
        .token_offsets = output->token_offsets
                             ? &output->token_offsets[*total_tokens]
                             : NULL,
        .type_ids = output->type_ids ? &output->type_ids[*total_tokens] : NULL,
    };
    iree_host_size_t tokens_written = 0;
    bool segment_complete = false;
    IREE_RETURN_IF_ERROR(iree_tokenizer_encode_ring_as_segment(
        state, /*is_partial=*/!force_complete, sub_output, &tokens_written,
        &segment_complete));
    iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor,
                                                 sub_output, 0, tokens_written);
    iree_tokenizer_postprocessor_trim_token_offsets(
        &state->postprocessor, state->tokenizer->vocab, sub_output, 0,
        tokens_written);
    *total_tokens += tokens_written;

    iree_host_size_t reclaimed =
        iree_tokenizer_model_state_reclaim(state->model_state);
    if (reclaimed > 0) {
      state->read_position += reclaimed;
      // Keep segmenter_view_start in sync: those bytes were processed by the
      // model directly, not through the segmenter. Without this, the segmenter
      // view (segmenter_view_start to write_position) would include reclaimed
      // bytes, producing segments with out-of-bounds positions.
      if (state->segmenter_view_start < state->read_position) {
        state->segmenter_view_start = state->read_position;
      }
    }
    // When a forced-complete drains the ring, exit partial mode so fresh
    // text after the special token goes through the normal segmenter path.
    if (force_complete && segment_complete &&
        state->read_position >= state->write_position) {
      state->has_partial_segment = false;
    }
    if (reclaimed > 0 || tokens_written > 0) {
      *out_made_progress = true;
    }
  }

  return iree_ok_status();
}

iree_status_t iree_tokenizer_encode_state_feed(
    iree_tokenizer_encode_state_t* state, iree_string_view_t chunk,
    iree_tokenizer_token_output_t output, iree_host_size_t* out_bytes_consumed,
    iree_host_size_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(output.token_ids);
  IREE_ASSERT_ARGUMENT(out_bytes_consumed);
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_bytes_consumed = 0;
  *out_token_count = 0;

  // Verify required pipeline stages are configured.
  // Note: segmenter_state may be NULL (passthrough mode).
  if (!state->model_state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "encode pipeline not fully configured");
  }

  iree_host_size_t original_chunk_size = chunk.size;
  iree_host_size_t total_tokens = 0;

  // Emit prefix special tokens before any model tokens.
  total_tokens += iree_tokenizer_postprocessor_emit_prefix(
      &state->postprocessor, output, total_tokens);

  // Pull-based processing: pump the pipeline until no more progress or error.
  bool made_progress = true;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && made_progress) {
    status = iree_tokenizer_encode_state_pump(state, &chunk, &output,
                                              &total_tokens, &made_progress);
  }

  *out_bytes_consumed = original_chunk_size - chunk.size;
  *out_token_count = total_tokens;

  // Detect deadlock: input remains but no progress was made. With partial
  // segment handling this should not happen in practice — the pump enters
  // partial mode when the ring is full with no segment boundaries, and BPE's
  // frozen token theorem guarantees progress. This can only fire if the ring
  // buffer is smaller than max_token_length (configuration error).
  if (iree_status_is_ok(status) && chunk.size > 0 && *out_bytes_consumed == 0 &&
      total_tokens == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "encode deadlock: no progress despite partial segment handling "
        "(logical_capacity=%" PRIhsz " bytes, used=%" PRIhsz
        " bytes, pending_input=%" PRIhsz " bytes, has_partial=%" PRIu32 ")",
        state->capacity_mask + 1, state->write_position - state->read_position,
        chunk.size, (uint32_t)state->has_partial_segment);
  }

  return status;
}

iree_status_t iree_tokenizer_encode_state_finalize(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(output.token_ids);
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_token_count = 0;

  // Mark finalize mode. Post-norm special token matching uses this to treat
  // NEED_MORE from speculative checks as NO_MATCH (no more input is coming,
  // so a partial prefix like "<" can never become a complete special token
  // like "<｜begin▁of▁sentence｜>").
  state->in_finalize_mode = true;

  // Verify required pipeline stages are configured.
  // Note: segmenter_state may be NULL (passthrough mode).
  if (!state->model_state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "encode pipeline not fully configured");
  }

  iree_host_size_t total_tokens = 0;

  // Emit any pending prefix tokens (handles the empty-input case where feed()
  // was never called and the state is still in PREFIX phase).
  total_tokens += iree_tokenizer_postprocessor_emit_prefix(
      &state->postprocessor, output, total_tokens);

  // Handle pending partial special token match from streaming input.
  // If the input ended with bytes that looked like they could be a special
  // token prefix but weren't completed, those bytes need to go to the
  // normalizer now (they're just regular text, not a special token).
  if (iree_tokenizer_special_tokens_encode_state_has_partial(
          &state->special_token_match)) {
    // Get the partial bytes.
    uint8_t partial_buffer[256];  // Max special token length is typically <30.
    iree_host_size_t partial_length =
        iree_tokenizer_special_tokens_encode_state_get_partial(
            &state->special_token_match, &state->tokenizer->special_tokens,
            partial_buffer);

    // Feed partial bytes to normalizer (or passthrough to ring).
    // Per CLAUDE.md "NO SILENT FAILURES": must not drop data without error.
    iree_host_size_t write_space = iree_tokenizer_ring_writable_space(state);
    if (write_space < partial_length) {
      // Ring buffer is full - cannot write partial bytes.
      // Return error instead of silently dropping data.
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "ring buffer full: need %" PRIhsz
                              " bytes, have %" PRIhsz,
                              partial_length, write_space);
    }
    iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
        state->write_position, state->capacity_mask);
    if (state->normalizer_state) {
      iree_string_view_t partial_view =
          iree_make_string_view((const char*)partial_buffer, partial_length);
      iree_mutable_string_view_t norm_output = {
          .data = (char*)state->transform_buffer.data + physical_write,
          .size = write_space,
      };
      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t bytes_written = 0;
      // Finalize: this is the end of the input stream, so signal segment end.
      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
          state->normalizer_state, partial_view, norm_output,
          IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END, &bytes_consumed,
          &bytes_written));
      iree_tokenizer_ring_fixup_write_wrap(state, physical_write,
                                           bytes_written);
      state->write_position += bytes_written;
    } else {
      // Passthrough: copy directly to ring.
      memcpy((char*)state->transform_buffer.data + physical_write,
             partial_buffer, partial_length);
      iree_tokenizer_ring_fixup_write_wrap(state, physical_write,
                                           partial_length);
      state->write_position += partial_length;
    }
    // Clear partial match state only after successful write.
    iree_tokenizer_special_tokens_encode_state_clear_partial(
        &state->special_token_match);
  }

  // Handle pending post-normalization partial special token match.
  // Unlike pre-norm matches, these bytes are already in the ring buffer (they
  // passed through the normalizer). If the input ended mid-match, it's not a
  // special token - clear the state so the segmenter processes them normally.
  if (iree_tokenizer_special_tokens_encode_state_has_partial(
          &state->special_token_match_post)) {
    iree_tokenizer_special_tokens_encode_state_clear_partial(
        &state->special_token_match_post);
  }

  // Finalize normalizer - flush any buffered data to ring.
  // Writes may extend into the mirror region and are fixed up afterward.
  if (state->normalizer_state) {
    iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
        state->write_position, state->capacity_mask);
    iree_host_size_t write_space = iree_tokenizer_ring_writable_space(state);
    iree_mutable_string_view_t norm_output = iree_make_mutable_string_view(
        (char*)state->transform_buffer.data + physical_write, write_space);
    iree_host_size_t bytes_written = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_finalize(
        state->normalizer_state, norm_output, &bytes_written));
    iree_tokenizer_ring_fixup_write_wrap(state, physical_write, bytes_written);
    state->write_position += bytes_written;
  }

  if (state->has_partial_segment) {
    // Partial segment mode: feed remaining ring data as a final non-partial
    // segment. BPE completes BYTE_LOOP, transitions to FLUSH, and emits all
    // remaining window tokens. Segmenter is bypassed (split=false produced no
    // boundaries, so there's nothing meaningful to finalize).
    if (state->write_position > state->read_position &&
        output.capacity > total_tokens) {
      iree_tokenizer_token_output_t sub_output = {
          .capacity = output.capacity - total_tokens,
          .token_ids = &output.token_ids[total_tokens],
          .token_offsets =
              output.token_offsets ? &output.token_offsets[total_tokens] : NULL,
          .type_ids = output.type_ids ? &output.type_ids[total_tokens] : NULL,
      };
      iree_host_size_t tokens_written = 0;
      bool segment_complete = false;
      IREE_RETURN_IF_ERROR(iree_tokenizer_encode_ring_as_segment(
          state, /*is_partial=*/false, sub_output, &tokens_written,
          &segment_complete));
      iree_tokenizer_postprocessor_assign_type_ids(
          &state->postprocessor, sub_output, 0, tokens_written);
      iree_tokenizer_postprocessor_trim_token_offsets(
          &state->postprocessor, state->tokenizer->vocab, sub_output, 0,
          tokens_written);
      total_tokens += tokens_written;
      // Only advance positions when the model fully completed the segment
      // (all bytes processed + all tokens emitted). If output filled
      // mid-segment (BYTE_LOOP or FLUSH), the ring data is still needed for
      // continuation on the next finalize call.
      if (segment_complete) {
        state->read_position = state->write_position;
        state->segmenter_view_start = state->write_position;
      }
    }
  } else {
    // Normal path: use pump() to drain pipeline, then finalize segmenter.
    //
    // Two-phase drain:
    // 1. After normalizer finalize, pump() matches post-norm special tokens
    //    and segments the ring data via segmenter_state_process().
    // 2. After segmenter finalize forces the final segment boundary, pump()
    //    processes those segments through the model.
    //
    // This reuses pump() logic instead of duplicating it, ensuring consistent
    // handling of post-norm tokens, pending_special_token emission, and
    // segment processing. The drain loops are bounded: with empty input, no
    // new data enters the pipeline, and finite ring + segment buffers mean
    // pump() must eventually stop making progress.

    // Phase 1: Drain to match post-norm tokens and segment ring data.
    // Bounded by: ring buffer size (segments produced) + segment batch size.
    iree_string_view_t empty_chunk = iree_make_string_view(NULL, 0);
    iree_tokenizer_token_output_t drain_output = {
        .capacity = output.capacity - total_tokens,
        .token_ids = &output.token_ids[total_tokens],
        .token_offsets =
            output.token_offsets ? &output.token_offsets[total_tokens] : NULL,
        .type_ids = output.type_ids ? &output.type_ids[total_tokens] : NULL,
    };
    iree_host_size_t drain_tokens = 0;
    bool made_progress;
    do {
      IREE_RETURN_IF_ERROR(iree_tokenizer_encode_state_pump(
          state, &empty_chunk, &drain_output, &drain_tokens, &made_progress));
    } while (made_progress && drain_output.capacity > drain_tokens);
    total_tokens += drain_tokens;

    // Finalize segmenter - force final segment boundary for any remaining
    // ring data that pump() couldn't segment (segmenter was waiting for more
    // input to determine the boundary).
    if (state->segmenter_state &&
        state->segment_count < state->tokenizer->segment_batch_size) {
      iree_string_view_t remaining_input =
          iree_tokenizer_ring_get_segment_view(state);

      iree_tokenizer_segment_output_t segment_output = {
          .capacity =
              state->tokenizer->segment_batch_size - state->segment_count,
          .values = &state->segments[state->segment_count],
      };
      iree_host_size_t segments_produced = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_finalize(
          state->segmenter_state, remaining_input, segment_output,
          &segments_produced));

      // Capture segmenter_view_start before mutation for offset adjustment.
      iree_host_size_t view_start = state->segmenter_view_start;

      for (iree_host_size_t i = 0; i < segments_produced; ++i) {
        state->segments[state->segment_count + i].start += view_start;
        state->segments[state->segment_count + i].end += view_start;
        IREE_ASSERT(state->segments[state->segment_count + i].start <
                    state->segments[state->segment_count + i].end);
      }
      state->segment_count += segments_produced;

      if (!iree_tokenizer_segmenter_state_has_pending(state->segmenter_state)) {
        state->segmenter_view_start = state->write_position;
      }
    }

    // Phase 2: Drain to process the final segments and emit pending tokens.
    // Bounded by: segment batch size + 1 (for pending_special_token).
    drain_output = (iree_tokenizer_token_output_t){
        .capacity = output.capacity - total_tokens,
        .token_ids = &output.token_ids[total_tokens],
        .token_offsets =
            output.token_offsets ? &output.token_offsets[total_tokens] : NULL,
        .type_ids = output.type_ids ? &output.type_ids[total_tokens] : NULL,
    };
    drain_tokens = 0;
    do {
      IREE_RETURN_IF_ERROR(iree_tokenizer_encode_state_pump(
          state, &empty_chunk, &drain_output, &drain_tokens, &made_progress));
    } while (made_progress && drain_output.capacity > drain_tokens);
    total_tokens += drain_tokens;
  }

  // Finalize model - handle any trailing state (e.g., partial BPE merges).
  if (state->model_state) {
    iree_tokenizer_token_output_t remaining_output = {
        .capacity = output.capacity - total_tokens,
        .token_ids = &output.token_ids[total_tokens],
        .token_offsets =
            output.token_offsets ? &output.token_offsets[total_tokens] : NULL,
        .type_ids = output.type_ids ? &output.type_ids[total_tokens] : NULL,
    };
    iree_host_size_t final_tokens = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_model_state_finalize(
        state->model_state, remaining_output, &final_tokens));
    iree_tokenizer_postprocessor_assign_type_ids(
        &state->postprocessor, remaining_output, 0, final_tokens);
    iree_tokenizer_postprocessor_trim_token_offsets(
        &state->postprocessor, state->tokenizer->vocab, remaining_output, 0,
        final_tokens);
    total_tokens += final_tokens;
  }

  // Emit suffix special tokens (e.g., [SEP], </s>) after all model tokens.
  iree_tokenizer_postprocessor_begin_suffix(&state->postprocessor);
  total_tokens += iree_tokenizer_postprocessor_emit_suffix(
      &state->postprocessor, output, total_tokens);

  *out_token_count = total_tokens;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Streaming Decode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_decode_state_calculate_size(
    const iree_tokenizer_t* tokenizer, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;

  // Pre-decoded fast path: no decoder state or string buffer needed.
  if (tokenizer->pre_decoded.slab) {
    *out_size = sizeof(iree_tokenizer_decode_state_t);
    return iree_ok_status();
  }

  iree_host_size_t decoder_state_size =
      tokenizer->decoder ? iree_tokenizer_decoder_state_size(tokenizer->decoder)
                         : 0;

  return IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_decode_state_t), out_size,
      IREE_STRUCT_FIELD_ALIGNED(decoder_state_size, uint8_t, iree_max_align_t,
                                NULL),
      IREE_STRUCT_FIELD(tokenizer->string_batch_size, iree_string_view_t,
                        NULL));
}

iree_status_t iree_tokenizer_decode_state_initialize(
    const iree_tokenizer_t* tokenizer, iree_tokenizer_decode_flags_t flags,
    iree_byte_span_t state_storage, iree_tokenizer_decode_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(state_storage.data);
  IREE_ASSERT_ARGUMENT(out_state);

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  iree_tokenizer_decode_state_t* state = NULL;

  // Pre-decoded fast path: minimal state (no decoder state, no string buffer).
  if (tokenizer->pre_decoded.slab) {
    if (state_storage.data_length < sizeof(iree_tokenizer_decode_state_t)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "state_storage too small: need %" PRIhsz " bytes, got %" PRIhsz,
          sizeof(iree_tokenizer_decode_state_t), state_storage.data_length);
    }
    if (iree_status_is_ok(status)) {
      state = (iree_tokenizer_decode_state_t*)state_storage.data;
      memset(state, 0, sizeof(*state));
      state->tokenizer = tokenizer;
      state->flags = flags;
      state->is_first_token = true;
      *out_state = state;
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Calculate layout with offsets. Must match calculate_size exactly.
  iree_host_size_t decoder_state_size =
      tokenizer->decoder ? iree_tokenizer_decoder_state_size(tokenizer->decoder)
                         : 0;

  iree_host_size_t total_size = 0;
  iree_host_size_t decoder_offset = 0;
  iree_host_size_t strings_offset = 0;
  status = IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_decode_state_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(decoder_state_size, uint8_t, iree_max_align_t,
                                &decoder_offset),
      IREE_STRUCT_FIELD(tokenizer->string_batch_size, iree_string_view_t,
                        &strings_offset));

  if (iree_status_is_ok(status) && state_storage.data_length < total_size) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "state_storage too small: need %" PRIhsz
                              " bytes, got %" PRIhsz,
                              total_size, state_storage.data_length);
  }

  // Initialize state struct at base of storage.
  if (iree_status_is_ok(status)) {
    uint8_t* base = (uint8_t*)state_storage.data;
    state = (iree_tokenizer_decode_state_t*)base;
    memset(state, 0, sizeof(*state));

    state->tokenizer = tokenizer;
    state->flags = flags;
    state->is_first_token = true;
    state->string_count = 0;
    state->strings_consumed = 0;
  }

  // Initialize decoder state at computed offset.
  if (iree_status_is_ok(status) && tokenizer->decoder) {
    uint8_t* base = (uint8_t*)state_storage.data;
    status = iree_tokenizer_decoder_state_initialize(
        tokenizer->decoder, base + decoder_offset, &state->decoder_state);
  }

  if (iree_status_is_ok(status)) {
    // Set up string buffer at computed offset.
    uint8_t* base = (uint8_t*)state_storage.data;
    state->string_buffer = (iree_string_view_t*)(base + strings_offset);
    *out_state = state;
  } else {
    iree_tokenizer_decode_state_deinitialize(state);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_tokenizer_decode_state_deinitialize(
    iree_tokenizer_decode_state_t* state) {
  if (!state) return;

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_state_deinitialize(state->decoder_state);

  memset(state, 0, sizeof(*state));
  IREE_TRACE_ZONE_END(z0);
}

// Two-stage pump for decode pipeline. Runs stages in reverse pipeline order
// (decoder first, then vocab lookup) for pull-based flow control.
//
// Stage 1: Feed pending strings from buffer to decoder, write text to output.
// Stage 2: Look up tokens from input, fill string buffer.
//
// Returns whether any progress was made (strings consumed, text written,
// or tokens looked up). Caller loops until no progress with input/output
// exhausted.
static iree_status_t iree_tokenizer_decode_state_pump(
    iree_tokenizer_decode_state_t* state,
    iree_tokenizer_token_id_list_t* tokens,
    iree_mutable_string_view_t* text_output, iree_host_size_t* total_written,
    bool* out_made_progress) {
  *out_made_progress = false;

  const iree_tokenizer_t* tokenizer = state->tokenizer;
  iree_host_size_t string_batch_size = tokenizer->string_batch_size;

  // Stage 1: Decoder - consume strings from buffer, produce text.
  iree_host_size_t pending_count =
      state->string_count - state->strings_consumed;
  if (pending_count > 0 &&
      text_output->size >= IREE_TOKENIZER_DECODER_MIN_BUFFER_SIZE) {
    iree_tokenizer_string_list_t pending_strings = {
        pending_count,
        &state->string_buffer[state->strings_consumed],
    };
    iree_host_size_t strings_consumed = 0;
    iree_host_size_t bytes_written = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_decoder_state_process(
        state->decoder_state, pending_strings, *text_output, &strings_consumed,
        &bytes_written));

    state->strings_consumed += strings_consumed;
    *total_written += bytes_written;
    text_output->data += bytes_written;
    text_output->size -= bytes_written;

    if (strings_consumed > 0 || bytes_written > 0) {
      *out_made_progress = true;
    }

    // Compact buffer if all strings consumed.
    if (state->strings_consumed >= state->string_count) {
      state->string_count = 0;
      state->strings_consumed = 0;
    }
  }

  // Stage 2: Vocab lookup - consume tokens from input, fill string buffer.
  // Special tokens are skipped entirely (no text produced) when
  // SKIP_SPECIAL_TOKENS is set.
  iree_host_size_t buffer_available = string_batch_size - state->string_count;
  if (buffer_available > 0 && tokens->count > 0) {
    const bool skip_special = iree_all_bits_set(
        state->flags, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS);
    iree_host_size_t max_scan =
        buffer_available < tokens->count ? buffer_available : tokens->count;
    iree_host_size_t consumed = 0;
    for (iree_host_size_t i = 0; i < max_scan; ++i) {
      iree_tokenizer_token_id_t id = tokens->values[i];
      if (skip_special && iree_any_bit_set(iree_tokenizer_vocab_token_attrs(
                                               tokenizer->vocab, id),
                                           IREE_TOKENIZER_TOKEN_ATTR_SPECIAL)) {
        ++consumed;
        continue;
      }
      state->string_buffer[state->string_count++] =
          iree_tokenizer_vocab_token_text(tokenizer->vocab, id);
      ++consumed;
    }
    tokens->values += consumed;
    tokens->count -= consumed;
    if (consumed > 0) {
      *out_made_progress = true;
    }
  }

  return iree_ok_status();
}

// Pre-decoded fast path: token IDs → memcpy from pre-computed table.
// Bypasses the entire decoder pipeline (no string buffer, no vtable dispatch).
// Copies up to 16 bytes using overlapping word loads/stores.
// Avoids memcpy PLT call overhead for short copies (typical text tokens are
// 3-8 bytes where function call overhead dominates the actual copy work).
static inline void iree_tokenizer_copy_short(uint8_t* IREE_RESTRICT dst,
                                             const uint8_t* IREE_RESTRICT src,
                                             uint32_t length) {
  if (length >= 8) {
    uint64_t w0, w1;
    memcpy(&w0, src, 8);
    memcpy(&w1, src + length - 8, 8);
    memcpy(dst, &w0, 8);
    memcpy(dst + length - 8, &w1, 8);
  } else if (length >= 4) {
    uint32_t w0, w1;
    memcpy(&w0, src, 4);
    memcpy(&w1, src + length - 4, 4);
    memcpy(dst, &w0, 4);
    memcpy(dst + length - 4, &w1, 4);
  } else if (length > 0) {
    dst[0] = src[0];
    if (length > 1) {
      uint16_t w;
      memcpy(&w, src + length - 2, 2);
      memcpy(dst + length - 2, &w, 2);
    }
  }
}

// Handles the first token in a position-sensitive decode stream.
// Strips the leading space from the first valid token's pre-decoded form.
// Skips special tokens when SKIP_SPECIAL_TOKENS flag is set.
// Returns the number of tokens consumed (including skipped out-of-range IDs)
// and writes the decoded bytes to |output|. Sets |*out_bytes_written| to the
// number of bytes written (0 if output was too small or no valid token found).
static iree_host_size_t iree_tokenizer_decode_pre_decoded_first_token(
    const uint32_t* IREE_RESTRICT offsets,
    const uint8_t* IREE_RESTRICT src_data, iree_host_size_t vocab_capacity,
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_decode_flags_t flags,
    iree_tokenizer_token_id_list_t tokens, uint8_t* IREE_RESTRICT output,
    iree_host_size_t output_capacity, iree_host_size_t* out_bytes_written) {
  *out_bytes_written = 0;
  const bool skip_special =
      iree_all_bits_set(flags, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS);
  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    iree_tokenizer_token_id_t id = tokens.values[i];
    if (IREE_UNLIKELY(id < 0 || (iree_host_size_t)id >= vocab_capacity)) {
      continue;
    }
    if (skip_special &&
        iree_any_bit_set(iree_tokenizer_vocab_token_attrs(vocab, id),
                         IREE_TOKENIZER_TOKEN_ATTR_SPECIAL)) {
      continue;
    }
    uint32_t start = offsets[id];
    uint32_t length = offsets[id + 1] - start;
    if (length > 0 && src_data[start] == ' ') {
      ++start;
      --length;
    }
    if (length > output_capacity) return i;  // Output full, token not consumed.
    memcpy(output, src_data + start, length);
    *out_bytes_written = length;
    return i + 1;
  }
  return tokens.count;
}

// Flushes pending bytes from the inline byte accumulator as U+FFFD replacement
// characters. Each pending byte becomes one replacement character (3 bytes).
// Returns bytes written, or 0 if output buffer is too small.
static iree_host_size_t iree_tokenizer_decode_flush_pending_as_replacement(
    iree_tokenizer_decode_state_t* state, uint8_t* output,
    iree_host_size_t position, iree_host_size_t output_capacity) {
  iree_host_size_t written = 0;
  while (state->byte_fallback_pending_count > 0) {
    if (position + written + 3 > output_capacity) break;
    int encoded_length = iree_unicode_utf8_encode(
        IREE_UNICODE_REPLACEMENT_CHAR, (char*)(output + position + written));
    if (encoded_length <= 0) break;
    written += (iree_host_size_t)encoded_length;
    // Shift pending bytes left.
    for (uint8_t j = 0; j < state->byte_fallback_pending_count - 1; ++j) {
      state->byte_fallback_pending[j] = state->byte_fallback_pending[j + 1];
    }
    state->byte_fallback_pending_count--;
  }
  if (state->byte_fallback_pending_count == 0) {
    state->byte_fallback_expected_length = 0;
  }
  return written;
}

// Hybrid pre-decoded path for ByteFallback models. Non-byte tokens use the
// O(1) memcpy fast path from the pre-decoded table. Byte tokens (<0xHH>) are
// handled by an inline UTF-8 accumulator that validates sequences and emits
// raw bytes or U+FFFD replacement characters.
//
// Byte tokens are identified by contiguous ID range check (O(1)).
// Their raw byte value is stored as 1 byte in the pre-decoded data table.
static iree_status_t
iree_tokenizer_decode_state_feed_pre_decoded_with_byte_fallback(
    iree_tokenizer_decode_state_t* state, iree_tokenizer_token_id_list_t tokens,
    iree_mutable_string_view_t text_output,
    iree_host_size_t* out_tokens_consumed, iree_host_size_t* out_text_length) {
  const iree_tokenizer_t* tokenizer = state->tokenizer;
  const iree_tokenizer_pre_decoded_t* pre_decoded = &tokenizer->pre_decoded;
  const uint32_t* IREE_RESTRICT offsets = pre_decoded->offsets;
  const uint8_t* IREE_RESTRICT src_data = pre_decoded->data;
  uint8_t* IREE_RESTRICT output = (uint8_t*)text_output.data;
  const iree_host_size_t output_capacity = text_output.size;
  const iree_host_size_t vocab_capacity =
      iree_tokenizer_vocab_capacity(tokenizer->vocab);
  const bool skip_special = iree_all_bits_set(
      state->flags, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS);
  const int32_t byte_first = pre_decoded->byte_token_first_id;
  const int32_t byte_last = pre_decoded->byte_token_last_id;
  const uint8_t* byte_bitmap = pre_decoded->byte_token_bitmap;

  iree_host_size_t i = 0;
  iree_host_size_t total_written = 0;

  // Handle position-sensitive first token outside the main loop.
  if (state->is_first_token && pre_decoded->position_sensitive) {
    iree_host_size_t first_bytes = 0;
    i = iree_tokenizer_decode_pre_decoded_first_token(
        offsets, src_data, vocab_capacity, tokenizer->vocab, state->flags,
        tokens, output, output_capacity, &first_bytes);
    total_written = first_bytes;
    if (first_bytes > 0) {
      state->is_first_token = false;
    }
  }

  while (i < tokens.count) {
    iree_tokenizer_token_id_t id = tokens.values[i];

    if (IREE_UNLIKELY(id < 0 || (iree_host_size_t)id >= vocab_capacity)) {
      ++i;
      continue;
    }

    if (skip_special &&
        iree_any_bit_set(iree_tokenizer_vocab_token_attrs(tokenizer->vocab, id),
                         IREE_TOKENIZER_TOKEN_ATTR_SPECIAL)) {
      ++i;
      continue;
    }

    if (id >= byte_first && id <= byte_last &&
        (byte_bitmap[(id - byte_first) / 8] &
         (1u << ((id - byte_first) % 8)))) {
      // Byte token: read raw byte value from pre-decoded table.
      uint8_t byte_value = src_data[offsets[id]];

      if (state->byte_fallback_pending_count == 0) {
        // No pending bytes — start a new UTF-8 sequence.
        iree_host_size_t sequence_length =
            iree_unicode_utf8_sequence_length(byte_value);

        if (sequence_length == 1 && (byte_value & 0x80) == 0) {
          // ASCII byte — emit directly.
          if (total_written + 1 > output_capacity) break;
          output[total_written++] = byte_value;
          ++i;
          continue;
        }

        if (sequence_length == 1) {
          // Invalid lead byte (continuation byte as lead) — emit U+FFFD.
          if (total_written + 3 > output_capacity) break;
          int encoded_length = iree_unicode_utf8_encode(
              IREE_UNICODE_REPLACEMENT_CHAR, (char*)(output + total_written));
          if (encoded_length > 0)
            total_written += (iree_host_size_t)encoded_length;
          ++i;
          continue;
        }

        // Multi-byte sequence — start accumulating.
        state->byte_fallback_pending[0] = byte_value;
        state->byte_fallback_pending_count = 1;
        state->byte_fallback_expected_length = (uint8_t)sequence_length;
        ++i;
        continue;
      }

      // Have pending bytes — expecting continuation.
      if ((byte_value & 0xC0) != 0x80) {
        // Not a continuation byte — flush pending as replacements, then
        // re-process this byte token (don't advance i).
        iree_host_size_t flushed =
            iree_tokenizer_decode_flush_pending_as_replacement(
                state, output, total_written, output_capacity);
        if (flushed == 0 && state->byte_fallback_pending_count > 0) break;
        total_written += flushed;
        // Don't advance i — re-process this byte as a new sequence start.
        continue;
      }

      // Valid continuation byte.
      state->byte_fallback_pending[state->byte_fallback_pending_count++] =
          byte_value;

      if (state->byte_fallback_pending_count ==
          state->byte_fallback_expected_length) {
        // Sequence complete — validate via utf8_decode.
        iree_string_view_t pending =
            iree_make_string_view((const char*)state->byte_fallback_pending,
                                  state->byte_fallback_pending_count);
        iree_host_size_t decode_position = 0;
        uint32_t codepoint =
            iree_unicode_utf8_decode(pending, &decode_position);

        bool is_valid =
            (decode_position == state->byte_fallback_pending_count &&
             codepoint != IREE_UNICODE_REPLACEMENT_CHAR);

        if (is_valid) {
          // Valid UTF-8 — emit accumulated bytes.
          if (total_written + state->byte_fallback_pending_count >
              output_capacity) {
            // Buffer full — undo last byte and stop.
            state->byte_fallback_pending_count--;
            break;
          }
          memcpy(output + total_written, state->byte_fallback_pending,
                 state->byte_fallback_pending_count);
          total_written += state->byte_fallback_pending_count;
          state->byte_fallback_pending_count = 0;
          state->byte_fallback_expected_length = 0;
        } else {
          // Invalid sequence — flush as replacements.
          iree_host_size_t flushed =
              iree_tokenizer_decode_flush_pending_as_replacement(
                  state, output, total_written, output_capacity);
          if (flushed == 0 && state->byte_fallback_pending_count > 0) {
            state->byte_fallback_pending_count--;
            break;
          }
          total_written += flushed;
        }
      }

      ++i;
    } else {
      // Non-byte token: flush any pending byte accumulator first.
      if (state->byte_fallback_pending_count > 0) {
        iree_host_size_t flushed =
            iree_tokenizer_decode_flush_pending_as_replacement(
                state, output, total_written, output_capacity);
        if (flushed == 0 && state->byte_fallback_pending_count > 0) break;
        total_written += flushed;
      }

      // Standard pre-decoded memcpy fast path.
      uint32_t start = offsets[id];
      uint32_t length = offsets[id + 1] - start;

      if (IREE_UNLIKELY(length > output_capacity - total_written)) {
        break;  // Token not consumed.
      }

      if (IREE_LIKELY(length <= 16)) {
        iree_tokenizer_copy_short(output + total_written, src_data + start,
                                  length);
      } else {
        memcpy(output + total_written, src_data + start, length);
      }
      total_written += length;
      ++i;
    }
  }

  *out_tokens_consumed = i;
  *out_text_length = total_written;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decode_state_feed_pre_decoded(
    iree_tokenizer_decode_state_t* state, iree_tokenizer_token_id_list_t tokens,
    iree_mutable_string_view_t text_output,
    iree_host_size_t* out_tokens_consumed, iree_host_size_t* out_text_length) {
  // Dispatch to hybrid path when byte fallback is active.
  if (state->tokenizer->pre_decoded.has_byte_fallback) {
    return iree_tokenizer_decode_state_feed_pre_decoded_with_byte_fallback(
        state, tokens, text_output, out_tokens_consumed, out_text_length);
  }

  const iree_tokenizer_t* tokenizer = state->tokenizer;
  const iree_tokenizer_pre_decoded_t* pre_decoded = &tokenizer->pre_decoded;
  const uint32_t* IREE_RESTRICT offsets = pre_decoded->offsets;
  const uint8_t* IREE_RESTRICT src_data = pre_decoded->data;
  uint8_t* IREE_RESTRICT output = (uint8_t*)text_output.data;
  const iree_host_size_t output_capacity = text_output.size;
  const iree_host_size_t vocab_capacity =
      iree_tokenizer_vocab_capacity(tokenizer->vocab);
  const bool skip_special = iree_all_bits_set(
      state->flags, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS);

  iree_host_size_t i = 0;
  iree_host_size_t total_written = 0;

  // Handle position-sensitive first token outside the main loop.
  // This eliminates a per-iteration branch check in the hot path below.
  if (state->is_first_token && pre_decoded->position_sensitive) {
    iree_host_size_t first_bytes = 0;
    i = iree_tokenizer_decode_pre_decoded_first_token(
        offsets, src_data, vocab_capacity, tokenizer->vocab, state->flags,
        tokens, output, output_capacity, &first_bytes);
    total_written = first_bytes;
    // Only clear is_first_token if we actually emitted text (found a valid
    // non-special token). If all tokens so far were skipped/invalid, the next
    // feed call's first valid token still needs space-stripping.
    if (first_bytes > 0) {
      state->is_first_token = false;
    }
  }

  // Main decode loop: no position-sensitive checks, inline small copies.
  while (i < tokens.count) {
    iree_tokenizer_token_id_t id = tokens.values[i];
    ++i;

    if (IREE_UNLIKELY(id < 0 || (iree_host_size_t)id >= vocab_capacity)) {
      continue;
    }

    if (skip_special &&
        iree_any_bit_set(iree_tokenizer_vocab_token_attrs(tokenizer->vocab, id),
                         IREE_TOKENIZER_TOKEN_ATTR_SPECIAL)) {
      continue;
    }

    uint32_t start = offsets[id];
    uint32_t length = offsets[id + 1] - start;

    if (IREE_UNLIKELY(length > output_capacity - total_written)) {
      --i;  // This token was not consumed.
      break;
    }

    if (IREE_LIKELY(length <= 16)) {
      iree_tokenizer_copy_short(output + total_written, src_data + start,
                                length);
    } else {
      memcpy(output + total_written, src_data + start, length);
    }
    total_written += length;
  }

  *out_tokens_consumed = i;
  *out_text_length = total_written;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_decode_state_feed(
    iree_tokenizer_decode_state_t* state, iree_tokenizer_token_id_list_t tokens,
    iree_mutable_string_view_t text_output,
    iree_host_size_t* out_tokens_consumed, iree_host_size_t* out_text_length) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(tokens.values || tokens.count == 0);
  IREE_ASSERT_ARGUMENT(text_output.data);
  IREE_ASSERT_ARGUMENT(out_tokens_consumed);
  IREE_ASSERT_ARGUMENT(out_text_length);
  *out_tokens_consumed = 0;
  *out_text_length = 0;

  if (state->tokenizer->pre_decoded.slab) {
    return iree_tokenizer_decode_state_feed_pre_decoded(
        state, tokens, text_output, out_tokens_consumed, out_text_length);
  }

  if (!state->decoder_state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decode pipeline not configured");
  }

  // Track original input count to compute consumption.
  iree_host_size_t original_count = tokens.count;
  iree_host_size_t total_written = 0;

  // Pump until no progress (input exhausted or output full).
  bool made_progress = true;
  while (made_progress) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_pump(
        state, &tokens, &text_output, &total_written, &made_progress));
  }

  *out_tokens_consumed = original_count - tokens.count;
  *out_text_length = total_written;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_decode_state_finalize(
    iree_tokenizer_decode_state_t* state,
    iree_mutable_string_view_t text_output, iree_host_size_t* out_text_length) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(text_output.data);
  IREE_ASSERT_ARGUMENT(out_text_length);
  *out_text_length = 0;

  // Pre-decoded fast path: flush any pending byte accumulator.
  if (state->tokenizer->pre_decoded.slab) {
    if (state->byte_fallback_pending_count > 0) {
      iree_host_size_t flushed =
          iree_tokenizer_decode_flush_pending_as_replacement(
              state, (uint8_t*)text_output.data, 0, text_output.size);
      *out_text_length = flushed;
    }
    return iree_ok_status();
  }

  if (!state->decoder_state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decode pipeline not configured");
  }

  return iree_tokenizer_decoder_state_finalize(state->decoder_state,
                                               text_output, out_text_length);
}
