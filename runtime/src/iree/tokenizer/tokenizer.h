// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_TOKENIZER_H_
#define IREE_TOKENIZER_TOKENIZER_H_

#include "iree/base/api.h"
#include "iree/tokenizer/postprocessor.h"
#include "iree/tokenizer/special_tokens.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Transform Buffer Configuration
//===----------------------------------------------------------------------===//

// Maximum transform buffer size for one-shot and streaming encode operations.
// This limits stack/heap allocation for the internal ring buffer used during
// normalization and segmentation. Larger buffers reduce the chance of needing
// emergency flush for pathological inputs (thousands of characters without
// punctuation), but use more memory.
//
// The transform buffer uses double-buffer mode, so the logical capacity is half
// the allocation. A 64KB allocation provides 32KB logical capacity.
//
// Maximum transform buffer allocation.
// Recommended values:
//   8KB:  Minimal memory, suitable for most English text
//  16KB:  Good balance for mixed content and cache locality
//
// The transform buffer determines the maximum segment size that the segmenter
// pipeline processes. Larger buffers allow longer segments, but segments that
// exceed L1 data cache (typically 32KB) cause DFA cache misses during
// multi-stage pre-tokenization. 16KB (8KB logical capacity after ring buffer
// halving) keeps the working set cache-friendly for segmenters with multiple
// children (e.g., DeepSeek V3's 4-stage Sequence pre-tokenizer).
#ifndef IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE
#define IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE (16 * 1024)
#endif

// Normalizer expansion factor for transform buffer sizing.
// The ring buffer's logical capacity (allocation / 2) must be large enough to
// hold the entire normalized output of a segment. Normalizers can expand input:
// the replace normalizer substitutes 1-byte spaces with 3-byte ▁ (U+2581),
// so worst-case expansion is 3× for space-heavy text. This multiplier is
// applied to the input text size before computing the buffer allocation.
//
// For models without expanding normalizers, the extra allocation is harmless
// (it's a transient per-encode allocation). Increase this if encountering
// RESOURCE_EXHAUSTED errors from streaming encode deadlocks.
#ifndef IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR
#define IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR 3
#endif

// Minimum transform buffer allocation returned by recommended_size().
// Below this threshold, the ring buffer's logical capacity (allocation / 2)
// is too small for the segmenter to reliably identify complete segment
// boundaries, causing artificial segment splits and sub-optimal BPE merges
// (allocation=256 → +8.5% tokens, allocation=1024 → +1.4% tokens,
// allocation=2048+ → optimal). The value 4096 (2048 bytes logical capacity)
// ensures optimal BPE quality for all practical inputs.
//
// The encode_state_initialize function accepts any power-of-two allocation
// >= 8. This constant only affects recommended_size(), not the hard minimum.
#define IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE 4096

// Tokenizer with builder pattern for construction.
// See README.md for architecture, streaming patterns, and design details.
//
// Usage:
//   // 1. Create builder (stack-allocated)
//   iree_tokenizer_builder_t builder;
//   iree_tokenizer_builder_initialize(allocator, &builder);
//
//   // 2. Format loader populates builder
//   format_loader_parse_json(&builder, json);  // transfers ownership
//
//   // 3. Build tokenizer (transfers ownership from builder)
//   iree_tokenizer_t* tokenizer = NULL;
//   IREE_RETURN_IF_ERROR(iree_tokenizer_builder_build(&builder, &tokenizer));
//
//   // 4. Cleanup builder (no-op after successful build)
//   iree_tokenizer_builder_deinitialize(&builder);
//
//   // 5. Use tokenizer for encode/decode
//   iree_tokenizer_encode(tokenizer, text, ...);
//
//   // 6. Free tokenizer when done
//   iree_tokenizer_free(tokenizer);
//
// Thread-safe, immutable.
typedef struct iree_tokenizer_t iree_tokenizer_t;

// Streaming encode state.
typedef struct iree_tokenizer_encode_state_t iree_tokenizer_encode_state_t;

// Streaming decode state.
typedef struct iree_tokenizer_decode_state_t iree_tokenizer_decode_state_t;

// Component forward declarations.
typedef struct iree_tokenizer_normalizer_t iree_tokenizer_normalizer_t;
typedef struct iree_tokenizer_segmenter_t iree_tokenizer_segmenter_t;
typedef struct iree_tokenizer_model_t iree_tokenizer_model_t;
typedef struct iree_tokenizer_decoder_t iree_tokenizer_decoder_t;
typedef struct iree_tokenizer_vocab_t iree_tokenizer_vocab_t;

//===----------------------------------------------------------------------===//
// Offset Run Tracking (Internal)
//===----------------------------------------------------------------------===//

// Run-length encoded offset mapping (internal).
// Used to track offset changes through length-changing transforms.
typedef struct iree_tokenizer_offset_run_t {
  iree_host_size_t transform_position;  // Position in transform buffer.
  iree_host_size_t original_offset;     // Corresponding original input offset.
} iree_tokenizer_offset_run_t;

// List of offset runs for tracking (pointer + capacity).
typedef struct iree_tokenizer_offset_run_list_t {
  iree_host_size_t capacity;
  iree_tokenizer_offset_run_t* values;
} iree_tokenizer_offset_run_list_t;

static inline iree_tokenizer_offset_run_list_t
iree_tokenizer_offset_run_list_empty(void) {
  iree_tokenizer_offset_run_list_t list = {0, NULL};
  return list;
}

//===----------------------------------------------------------------------===//
// Encode Flags
//===----------------------------------------------------------------------===//

// Flags for encode state initialization.
enum iree_tokenizer_encode_flag_bits_e {
  IREE_TOKENIZER_ENCODE_FLAG_NONE = 0u,
  // Indicates this is the start of input (for Metaspace "First" prepend mode).
  IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START = 1u << 0,
  // Request offset tracking (requires offset_runs storage at initialize).
  IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS = 1u << 1,
  // Insert special tokens from the postprocessor template (prefix, suffix).
  // When set, the postprocessor's active template (single or pair) is used
  // to emit special tokens and assign type_ids to model-produced tokens.
  // When not set, no special tokens are inserted and type_ids are all 0.
  IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS = 1u << 2,
};
typedef uint32_t iree_tokenizer_encode_flags_t;

//===----------------------------------------------------------------------===//
// Decode Flags
//===----------------------------------------------------------------------===//

// Flags for decode state initialization.
enum iree_tokenizer_decode_flag_bits_e {
  IREE_TOKENIZER_DECODE_FLAG_NONE = 0u,
  // Skip tokens with the SPECIAL attribute (BOS/EOS/CLS/SEP/PAD/UNK).
  // When set, special tokens produce no output text. This matches the default
  // behavior of HuggingFace's tokenizer.decode(skip_special_tokens=True).
  // When not set, all tokens are decoded including special tokens.
  IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS = 1u << 0,
};
typedef uint32_t iree_tokenizer_decode_flags_t;

//===----------------------------------------------------------------------===//
// Builder (Stack-Allocatable)
//===----------------------------------------------------------------------===//

// Builder for constructing tokenizers.
// Stack-allocated by caller, accumulates heap allocations during loading,
// then transfers ownership to tokenizer via build().
//
// Lifecycle:
//   initialize() -> set_*() ... -> build() -> deinitialize()
//
// After successful build(), deinitialize() is a no-op (ownership transferred).
// On error, deinitialize() frees any partial state.
typedef struct iree_tokenizer_builder_t {
  iree_allocator_t allocator;

  // Pipeline components (owned by builder until build() transfers them).
  // Each set_*() method transfers ownership to the builder.
  // Optional - NULL for no-op normalizer.
  iree_tokenizer_normalizer_t* normalizer;
  iree_tokenizer_segmenter_t* segmenter;  // Required.
  iree_tokenizer_model_t* model;          // Required.
  // Optional - NULL for no-op decode.
  iree_tokenizer_decoder_t* decoder;

  // Post-processor (value type, embedded directly). When all template counts
  // are zero and both flags are false (the zero-initialized state), runtime
  // postprocessor operations are no-ops.
  iree_tokenizer_postprocessor_t postprocessor;

  // Special tokens for pre-normalization matching (value type, embedded).
  // When empty (zero-initialized), no special token matching is performed.
  iree_tokenizer_special_tokens_t special_tokens;

  // Special tokens for post-normalization matching (value type, embedded).
  // These have normalized=true and are matched after the normalizer runs.
  // When empty (zero-initialized), no post-norm matching is performed.
  iree_tokenizer_special_tokens_t special_tokens_post_norm;

  // Vocabulary (owned by builder until build() transfers it).
  // The model references but does not own the vocab, so vocab must be stored
  // separately and freed after the model.
  iree_tokenizer_vocab_t* vocab;

  // Number of token strings to batch during decode vocab lookups.
  // Larger values amortize vtable overhead but use more state storage.
  // Initialized to IREE_TOKENIZER_DEFAULT_STRING_BATCH_SIZE.
  iree_host_size_t string_batch_size;
} iree_tokenizer_builder_t;

// Initializes a builder (stack-allocated by caller).
void iree_tokenizer_builder_initialize(iree_allocator_t allocator,
                                       iree_tokenizer_builder_t* out_builder);

// Cleans up builder state.
// No-op if build() succeeded (ownership transferred).
// Frees partial state if build() was never called or failed.
void iree_tokenizer_builder_deinitialize(iree_tokenizer_builder_t* builder);

// Sets the normalizer for the tokenizer. Transfers ownership to the builder.
// Optional - if not set, input is copied directly to the transform buffer.
void iree_tokenizer_builder_set_normalizer(
    iree_tokenizer_builder_t* builder, iree_tokenizer_normalizer_t* normalizer);

// Sets the segmenter for the tokenizer. Transfers ownership to the builder.
// Required - must be set before build().
void iree_tokenizer_builder_set_segmenter(
    iree_tokenizer_builder_t* builder, iree_tokenizer_segmenter_t* segmenter);

// Sets the model for the tokenizer. Transfers ownership to the builder.
// Required - must be set before build().
void iree_tokenizer_builder_set_model(iree_tokenizer_builder_t* builder,
                                      iree_tokenizer_model_t* model);

// Sets the decoder for the tokenizer. Transfers ownership to the builder.
// Optional - if not set, decode operations will fail.
void iree_tokenizer_builder_set_decoder(iree_tokenizer_builder_t* builder,
                                        iree_tokenizer_decoder_t* decoder);

// Sets the post-processor for the tokenizer. Copies the value into the builder.
// Optional - if not set (zero-initialized), postprocessor operations are no-ops
// and no special tokens are inserted during encoding.
void iree_tokenizer_builder_set_postprocessor(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_postprocessor_t postprocessor);

// Sets the special tokens collection for pre-normalization matching.
// Moves ownership of the collection's slab into the builder. After this call,
// |special_tokens| is left in an empty state and should not be deinitialized.
// Optional - if not set (zero-initialized), no special token matching occurs.
void iree_tokenizer_builder_set_special_tokens(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_special_tokens_t* special_tokens);

// Sets the special tokens collection for post-normalization matching.
// These tokens have normalized=true in HuggingFace and are matched after the
// normalizer transforms the input. Moves ownership of the collection's slab
// into the builder. After this call, |special_tokens| is left in an empty
// state and should not be deinitialized.
// Optional - if not set (zero-initialized), no post-norm matching occurs.
void iree_tokenizer_builder_set_special_tokens_post_norm(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_special_tokens_t* special_tokens);

// Sets the vocabulary for the tokenizer. Transfers ownership to the builder.
// Required when using models that need vocabulary lookup (BPE, WordPiece, etc).
// The vocab must outlive the model as the model references but does not own it.
void iree_tokenizer_builder_set_vocab(iree_tokenizer_builder_t* builder,
                                      iree_tokenizer_vocab_t* vocab);

// Sets the string batch size for decode operations.
// This controls how many token strings are looked up from vocab before feeding
// to the decoder. Larger values (up to a few hundred) improve throughput by
// amortizing vtable call overhead.
void iree_tokenizer_builder_set_string_batch_size(
    iree_tokenizer_builder_t* builder, iree_host_size_t batch_size);

// Builds a tokenizer from the builder.
// Transfers ownership of all components from builder to tokenizer.
// Builder becomes empty after successful build.
iree_status_t iree_tokenizer_builder_build(iree_tokenizer_builder_t* builder,
                                           iree_tokenizer_t** out_tokenizer);

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

// Frees a tokenizer and all owned components.
void iree_tokenizer_free(iree_tokenizer_t* tokenizer);

// Returns the vocabulary owned by the tokenizer.
// The returned pointer is valid for the lifetime of the tokenizer.
const iree_tokenizer_vocab_t* iree_tokenizer_vocab(
    const iree_tokenizer_t* tokenizer);

// Returns the human-readable model type name (e.g., "BPE").
// The returned string_view is valid for the lifetime of the tokenizer.
iree_string_view_t iree_tokenizer_model_type_name(
    const iree_tokenizer_t* tokenizer);

//===----------------------------------------------------------------------===//
// Batch Encode/Decode (Convenience)
//===----------------------------------------------------------------------===//

// Encodes |text| to token IDs using |tokenizer|. This is a convenience wrapper
// around the streaming encode API equivalent to: initialize -> feed(text) ->
// finalize -> deinitialize.
//
// |flags| controls encoding behavior:
//   IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS: wraps output with
//     prefix/suffix special tokens (BOS, EOS, CLS, SEP) per the postprocessor.
// AT_INPUT_START and TRACK_OFFSETS (when output.token_offsets is non-NULL) are
// set automatically.
//
// Tokens are written to |output|. If output.token_offsets is non-NULL, byte
// ranges mapping each token back to the original input are also written. The
// actual number of tokens produced is returned in |out_token_count|.
//
// |allocator| is used for temporary state storage during encoding and is
// released before return.
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the output would exceed
// output.capacity. Callers can retry with a larger buffer.
iree_status_t iree_tokenizer_encode(const iree_tokenizer_t* tokenizer,
                                    iree_string_view_t text,
                                    iree_tokenizer_encode_flags_t flags,
                                    iree_tokenizer_token_output_t output,
                                    iree_allocator_t allocator,
                                    iree_host_size_t* out_token_count);

// Decodes |tokens| to text using |tokenizer|.
//
// |flags| controls decode behavior. SKIP_SPECIAL_TOKENS causes tokens with the
// SPECIAL attribute to produce no output (matching HuggingFace's default).
//
// Up to |text_output.size| bytes will be written to |text_output.data|. The
// output is raw UTF-8 bytes and is not NUL-terminated. The actual number of
// bytes produced is returned in |out_text_length|.
//
// |allocator| is used for temporary state storage during decoding and is
// released before return.
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the output would exceed
// text_output.size. Callers can retry with a larger buffer.
iree_status_t iree_tokenizer_decode(const iree_tokenizer_t* tokenizer,
                                    iree_tokenizer_token_id_list_t tokens,
                                    iree_tokenizer_decode_flags_t flags,
                                    iree_mutable_string_view_t text_output,
                                    iree_allocator_t allocator,
                                    iree_host_size_t* out_text_length);

//===----------------------------------------------------------------------===//
// Multi-Item Batch Encode/Decode
//===----------------------------------------------------------------------===//

// Input/output item for multi-item batch encoding. Caller allocates arrays of
// these items, fills in the inputs (text, output buffers), and the batch
// function fills in out_token_count for each.
typedef struct iree_tokenizer_encode_batch_item_t {
  // Input: text to encode.
  iree_string_view_t text;
  // Input: output buffers for token IDs and optional offsets.
  iree_tokenizer_token_output_t output;
  // Output: actual number of tokens written.
  iree_host_size_t out_token_count;
} iree_tokenizer_encode_batch_item_t;

// Encodes multiple text items in a single call, reusing internal state and
// scratch buffers across items for efficiency.
//
// |flags| controls encoding behavior (same as iree_tokenizer_encode).
// AT_INPUT_START and TRACK_OFFSETS are set automatically per-item.
//
// Caller provides |state_storage| (at least encode_state_calculate_size bytes)
// and |transform_buffer| for scratch space. For offset tracking across all
// items, provide |offset_runs| storage; otherwise pass an empty list.
//
// Each item in |items| must have valid text and output.token_ids pointer. If
// any item's output.token_offsets is non-NULL, offset tracking is enabled for
// that item (requires offset_runs to have sufficient capacity).
//
// On success, each item's out_token_count is set to the number of tokens
// produced. Returns IREE_STATUS_RESOURCE_EXHAUSTED if any item's output would
// exceed its output.capacity; in this case, processing stops at the failing
// item and prior items retain their results.
iree_status_t iree_tokenizer_encode_batch(
    const iree_tokenizer_t* tokenizer,
    iree_tokenizer_encode_batch_item_t* items, iree_host_size_t item_count,
    iree_tokenizer_encode_flags_t flags, iree_byte_span_t state_storage,
    iree_byte_span_t transform_buffer,
    iree_tokenizer_offset_run_list_t offset_runs);

// Input/output item for multi-item batch decoding. Caller allocates arrays of
// these items, fills in the inputs (tokens, text_output), and the batch
// function fills in out_text_length for each.
typedef struct iree_tokenizer_decode_batch_item_t {
  // Input: token IDs to decode.
  iree_tokenizer_token_id_list_t tokens;
  // Input: output buffer for text.
  iree_mutable_string_view_t text_output;
  // Output: actual number of bytes written.
  iree_host_size_t out_text_length;
} iree_tokenizer_decode_batch_item_t;

// Decodes multiple token sequences in a single call, reusing internal state
// across items for efficiency.
//
// |flags| controls decode behavior (same as iree_tokenizer_decode).
//
// Caller provides |state_storage| (at least decode_state_calculate_size bytes).
// Each item in |items| must have valid tokens (or empty if no tokens) and a
// valid text_output buffer with capacity in text_output.size.
//
// On success, each item's out_text_length is set to the number of bytes
// produced. Returns IREE_STATUS_RESOURCE_EXHAUSTED if any item's output would
// exceed its text_output.size; in this case, processing stops at the failing
// item and prior items retain their results.
iree_status_t iree_tokenizer_decode_batch(
    const iree_tokenizer_t* tokenizer,
    iree_tokenizer_decode_batch_item_t* items, iree_host_size_t item_count,
    iree_tokenizer_decode_flags_t flags, iree_byte_span_t state_storage);

//===----------------------------------------------------------------------===//
// Streaming Encode
//===----------------------------------------------------------------------===//

// Calculates the number of bytes required for encode state storage. Callers
// must allocate at least this many bytes and pass them to
// iree_tokenizer_encode_state_initialize.
// Returns IREE_STATUS_OUT_OF_RANGE on overflow.
iree_status_t iree_tokenizer_encode_state_calculate_size(
    const iree_tokenizer_t* tokenizer, iree_host_size_t* out_size);

// Calculates the recommended transform buffer size for encoding text.
// The returned size is guaranteed to be:
//   - A power of two (required by ring buffer mode)
//   - At least IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE
//   - At most IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE
//
// The text size is multiplied by
// IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR to account for normalizer
// expansion (e.g., space → ▁ is 1→3 bytes). The ring buffer's logical capacity
// is allocation/2, so a 3× multiplier yields 1.5× usable capacity relative to
// input — sufficient for typical text where <50% of bytes are subject to
// expansion.
//
// For streaming encode with multiple feed() calls, use the max expected chunk
// size as |text_size|. For one-shot encode, use the full text length.
static inline iree_host_size_t iree_tokenizer_transform_buffer_recommended_size(
    iree_host_size_t text_size) {
  // Account for normalizer expansion (saturating multiply to avoid overflow).
  iree_host_size_t expanded =
      text_size <= IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE /
                       IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR
          ? text_size * IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR
          : IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE;
  // Clamp to minimum.
  if (expanded < IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE) {
    expanded = IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE;
  }
  // Round up to next power of two if not already.
  iree_host_size_t size = iree_host_size_next_power_of_two(expanded);
  // Clamp to maximum.
  if (size > IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE) {
    size = IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE;
  }
  return size;
}

// Calculates the transform buffer size for one-shot (batch) encoding.
// Unlike the streaming recommended_size (which caps at 64KB for bounded
// memory), one-shot encoding uses a buffer proportional to the full input since
// it already allocates O(text_size) for output tokens. This avoids deadlocks
// when pre-tokenizer segments span more than the streaming buffer cap.
static inline iree_host_size_t iree_tokenizer_transform_buffer_oneshot_size(
    iree_host_size_t text_size) {
  iree_host_size_t expanded =
      text_size <= SIZE_MAX / IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR
          ? text_size * IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR
          : SIZE_MAX;
  if (expanded < IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE) {
    expanded = IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE;
  }
  return iree_host_size_next_power_of_two(expanded);
}

// Initializes encode state for streaming tokenization. The |tokenizer| must
// outlive the state.
//
// Caller provides |state_storage| (at least calculate_size bytes) and
// |transform_buffer| for scratch space during normalization. For offset
// tracking, provide |offset_runs| storage and set
// IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS in |flags|; otherwise pass an empty
// list.
//
// The initialized state is returned in |out_state|. The state does not own the
// storage buffers; the caller is responsible for keeping them valid until
// deinitialize is called.
iree_status_t iree_tokenizer_encode_state_initialize(
    const iree_tokenizer_t* tokenizer, iree_byte_span_t state_storage,
    iree_byte_span_t transform_buffer,
    iree_tokenizer_offset_run_list_t offset_runs,
    iree_tokenizer_encode_flags_t flags,
    iree_tokenizer_encode_state_t** out_state);

// Cleans up encode state. Does not free |state_storage| or other buffers
// provided at initialization as the caller owns them.
void iree_tokenizer_encode_state_deinitialize(
    iree_tokenizer_encode_state_t* state);

// Resets encode state for reuse with a new input, preserving the existing
// storage layout. This is more efficient than deinitialize+initialize when
// processing multiple inputs with the same tokenizer and buffers.
//
// |flags| can change between resets (e.g., different offset tracking per item).
// The transform_buffer and offset_runs remain as set during initialization.
void iree_tokenizer_encode_state_reset(iree_tokenizer_encode_state_t* state,
                                       iree_tokenizer_encode_flags_t flags);

// Returns true if the encode state has pending data that would produce tokens
// on finalize. Useful for determining whether to call finalize or whether the
// state is "clean" after processing. This is a lightweight query that does not
// modify state.
bool iree_tokenizer_encode_state_has_pending(
    const iree_tokenizer_encode_state_t* state);

// Feeds an input |chunk| to the encoder. Tokens that can be definitively
// produced are written to |output|. Additional tokens may be produced by
// subsequent feed calls or by finalize once all input has been provided.
//
// Pull-based model: Output buffer capacity drives processing. The encoder
// pulls data through the pipeline (normalizer → segmenter → model) to
// fill the output buffer. Processing stops when output is full or input
// is exhausted.
//
// Returns:
// - |out_bytes_consumed|: Bytes consumed from |chunk|. May be less than
//   chunk.size if internal buffers (transform_buffer) fill up. Caller should
//   retry with remaining bytes: chunk.data + *out_bytes_consumed.
// - |out_token_count|: Tokens written to |output|.
//
// The function always makes progress if possible: it will consume input
// and/or produce tokens. Returns iree_ok_status() even if not all input
// was consumed (check out_bytes_consumed).
iree_status_t iree_tokenizer_encode_state_feed(
    iree_tokenizer_encode_state_t* state, iree_string_view_t chunk,
    iree_tokenizer_token_output_t output, iree_host_size_t* out_bytes_consumed,
    iree_host_size_t* out_token_count);

// Finalizes encoding by flushing any buffered data through the pipeline. Must
// be called after all input chunks have been fed. Any remaining tokens are
// written to |output|.
//
// Returns the number of tokens written in |out_token_count|. Returns
// IREE_STATUS_RESOURCE_EXHAUSTED if the output would exceed output.capacity.
iree_status_t iree_tokenizer_encode_state_finalize(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* out_token_count);

//===----------------------------------------------------------------------===//
// Streaming Decode
//===----------------------------------------------------------------------===//

// Recommended output buffer size for decode_state_feed calls. Buffers of at
// least this size achieve full decode throughput. Smaller buffers are valid
// (minimum 4 bytes per IREE_TOKENIZER_DECODER_MIN_BUFFER_SIZE) but incur
// per-call overhead that reduces throughput by up to 12% at 64 bytes.
#define IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE 2048

// Calculates the number of bytes required for decode state storage. Callers
// must allocate at least this many bytes and pass them to
// iree_tokenizer_decode_state_initialize.
// Returns IREE_STATUS_OUT_OF_RANGE on overflow.
iree_status_t iree_tokenizer_decode_state_calculate_size(
    const iree_tokenizer_t* tokenizer, iree_host_size_t* out_size);

// Initializes decode state for streaming detokenization. The |tokenizer| must
// outlive the state.
//
// |flags| controls decode behavior (e.g., SKIP_SPECIAL_TOKENS). Flags are
// stored in the state and apply to all subsequent feed() calls.
//
// Caller provides |state_storage| (at least calculate_size bytes). The
// initialized state is returned in |out_state|. The state does not own the
// storage buffer; the caller is responsible for keeping it valid until
// deinitialize is called.
iree_status_t iree_tokenizer_decode_state_initialize(
    const iree_tokenizer_t* tokenizer, iree_tokenizer_decode_flags_t flags,
    iree_byte_span_t state_storage, iree_tokenizer_decode_state_t** out_state);

// Cleans up decode state. Does not free |state_storage| provided at
// initialization as the caller owns it.
void iree_tokenizer_decode_state_deinitialize(
    iree_tokenizer_decode_state_t* state);

// Feeds |tokens| to the decoder. Text that can be definitively produced is
// written to |text_output| as raw UTF-8 bytes (not NUL-terminated). Additional
// text may be produced by subsequent feed calls or by finalize once all tokens
// have been provided.
//
// Pull-based model: Output buffer capacity drives processing. The decoder
// pulls tokens through the pipeline to fill the text output. Processing
// stops when output is full or tokens are exhausted.
//
// Returns iree_ok_status() always (errors are only from internal pipeline
// failures). Progress is indicated by the output parameters:
// - |out_tokens_consumed|: Tokens consumed from |tokens|. May be less than
//   tokens.count if text output fills up. Caller should retry with remaining
//   tokens: tokens.values + *out_tokens_consumed.
// - |out_text_length|: Bytes written to |text_output|.
//
// Zero-progress case: If both out_tokens_consumed and out_text_length are 0,
// the output buffer is genuinely exhausted (the next token's text does not fit
// in |text_output|). Callers must check for this to avoid infinite loops.
// Provide a larger output buffer or consume the already-written text.
iree_status_t iree_tokenizer_decode_state_feed(
    iree_tokenizer_decode_state_t* state, iree_tokenizer_token_id_list_t tokens,
    iree_mutable_string_view_t text_output,
    iree_host_size_t* out_tokens_consumed, iree_host_size_t* out_text_length);

// Finalizes decoding by flushing any buffered data through the pipeline. Must
// be called after all tokens have been fed. Any remaining text is written to
// |text_output| as raw UTF-8 bytes (not NUL-terminated).
//
// Returns the number of bytes written in |out_text_length|. Returns
// IREE_STATUS_RESOURCE_EXHAUSTED if the output would exceed text_output.size.
iree_status_t iree_tokenizer_decode_state_finalize(
    iree_tokenizer_decode_state_t* state,
    iree_mutable_string_view_t text_output, iree_host_size_t* out_text_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TOKENIZER_H_
