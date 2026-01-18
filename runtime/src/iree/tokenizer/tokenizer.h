// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tokenizer interface for text encoding/decoding.
//
// This provides a unified interface for tokenizing text using different
// algorithms (BPE, WordPiece). The algorithm implementation is selected
// at tokenizer creation time via algorithm-specific allocators:
//
//   // For BPE (GPT-2, Llama, etc):
//   iree_tokenizer_bpe_allocate(vocab, allocator, &tokenizer);
//
//   // For WordPiece (BERT, etc):
//   iree_tokenizer_wordpiece_allocate(vocab, config, allocator, &tokenizer);
//
// Once created, all tokenizers use the same encode/decode/free API.

#ifndef IREE_TOKENIZER_TOKENIZER_H_
#define IREE_TOKENIZER_TOKENIZER_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"
#include "iree/tokenizer/literals.h"
#include "iree/tokenizer/postprocessor.h"
#include "iree/tokenizer/regex/exec.h"
#include "iree/tokenizer/transforms/transform.h"
#include "iree/tokenizer/types.h"
#include "iree/tokenizer/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Encoding Options
//===----------------------------------------------------------------------===//

// Flags controlling the encoding process.
typedef enum iree_tokenizer_encode_flag_bits_e {
  IREE_TOKENIZER_ENCODE_FLAG_DEFAULT = 0,
  // Add BOS/EOS/CLS/SEP as appropriate.
  IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS = 1 << 0,
  // Truncate from left instead of right.
  IREE_TOKENIZER_ENCODE_FLAG_TRUNCATE_LEFT = 1 << 1,
} iree_tokenizer_encode_flag_bits_t;
typedef uint32_t iree_tokenizer_encode_flags_t;

// Options controlling the encoding process.
typedef struct iree_tokenizer_encode_options_t {
  // Encoding behavior flags.
  iree_tokenizer_encode_flags_t flags;
  // Truncate to this length (0 = no limit).
  iree_host_size_t max_length;
} iree_tokenizer_encode_options_t;

// Default encoding options: add special tokens, no truncation.
#define IREE_TOKENIZER_ENCODE_OPTIONS_DEFAULT                 \
  ((iree_tokenizer_encode_options_t){                         \
      .flags = IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS, \
      .max_length = 0,                                        \
  })

//===----------------------------------------------------------------------===//
// Decoding Options
//===----------------------------------------------------------------------===//

// Flags controlling the decoding process.
typedef enum iree_tokenizer_decode_flag_bits_e {
  IREE_TOKENIZER_DECODE_FLAG_DEFAULT = 0,
  // Omit special tokens (BOS/EOS/CLS/SEP/etc) from output.
  IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS = 1 << 0,
} iree_tokenizer_decode_flag_bits_t;
typedef uint32_t iree_tokenizer_decode_flags_t;

//===----------------------------------------------------------------------===//
// Tokenizer Interface
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_t iree_tokenizer_t;

// Vtable for tokenizer implementations.
// Algorithm-specific allocators (bpe, wordpiece) populate this.
typedef struct iree_tokenizer_vtable_t {
  // Destroys the tokenizer and any algorithm-specific state.
  // Called by iree_tokenizer_free() to clean up the derived type.
  void (*destroy)(iree_tokenizer_t* tokenizer);

  // Encodes a single pre-tokenized word into token IDs.
  // This is the algorithm-specific part of encoding.
  iree_status_t (*encode_word)(const iree_tokenizer_t* tokenizer,
                               iree_string_view_t word, int32_t* out_ids,
                               iree_host_size_t max_ids,
                               iree_host_size_t* out_count);
} iree_tokenizer_vtable_t;

// Base tokenizer structure. Algorithm implementations extend this.
// Created via algorithm-specific allocators (iree_tokenizer_bpe_allocate, etc).
struct iree_tokenizer_t {
  const iree_tokenizer_vtable_t* vtable;
  iree_allocator_t allocator;
  const iree_tokenizer_vocab_t* vocab;  // Owned; freed by iree_tokenizer_free.
  iree_tokenizer_text_transform_t
      transform;                     // Pre-tokenizer (includes normalizer).
  iree_tokenizer_decoder_t decoder;  // Token decoder (for decode).
  iree_tokenizer_postprocessor_t postprocessor;  // Special token insertion.
  iree_tokenizer_literals_t literals;  // Literal tokens (added_tokens).
};

// Initializes the base tokenizer fields.
// Called by algorithm-specific allocators.
// |transform| is the pre-tokenizer transform (NULL for none). The transform
//             includes an optional normalizer that is applied inline.
// |decoder| is the token decoder (NULL for none).
// |postprocessor| is the special token inserter (NULL for none).
void iree_tokenizer_initialize(
    iree_tokenizer_t* tokenizer, const iree_tokenizer_vtable_t* vtable,
    iree_allocator_t allocator, const iree_tokenizer_vocab_t* vocab,
    const iree_tokenizer_text_transform_t* transform,
    const iree_tokenizer_decoder_t* decoder,
    const iree_tokenizer_postprocessor_t* postprocessor);

// Frees a tokenizer and all owned resources (including the vocab).
//
// After this call, the tokenizer pointer and any vocab pointer obtained from
// it are invalid. This is the only correct way to dispose of a tokenizer.
IREE_API_EXPORT void iree_tokenizer_free(iree_tokenizer_t* tokenizer);

// Returns the vocabulary used by this tokenizer.
IREE_API_EXPORT const iree_tokenizer_vocab_t* iree_tokenizer_vocab(
    const iree_tokenizer_t* tokenizer);

// Encodes text into token IDs via streaming callback.
//
// This is the base streaming API that emits token IDs in batches via callback.
// Use this for processing arbitrarily large inputs without buffer limits.
//
// |tokenizer| is the tokenizer to use.
// |text| is the input text (UTF-8).
// |flags| controls special token handling (ADD_SPECIAL_TOKENS).
//         Note: truncation flags are ignored in streaming mode.
// |callback| is invoked with batched token IDs.
// |user_data| is passed to the callback.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - Status from callback if it returns non-OK
//   - IREE_STATUS_INVALID_ARGUMENT for invalid inputs
IREE_API_EXPORT iree_status_t iree_tokenizer_encode_streaming(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_tokenizer_encode_flags_t flags,
    iree_tokenizer_token_callback_fn_t callback, void* user_data);

// Encodes text into token IDs (buffer-based wrapper).
//
// This is a convenience wrapper around iree_tokenizer_encode_streaming that
// collects tokens into a caller-provided buffer.
//
// |tokenizer| is the tokenizer to use.
// |text| is the input text (UTF-8).
// |options| controls special token handling and truncation.
// |out_ids| receives the token IDs (caller-allocated).
// |max_ids| is the capacity of out_ids.
// |out_count| receives the number of tokens produced.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_RESOURCE_EXHAUSTED if output buffer too small
//   - IREE_STATUS_INVALID_ARGUMENT for invalid inputs
IREE_API_EXPORT iree_status_t iree_tokenizer_encode(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_tokenizer_encode_options_t options, int32_t* out_ids,
    iree_host_size_t max_ids, iree_host_size_t* out_count);

// Decodes token IDs to text via streaming callback.
//
// This is the base streaming API that emits decoded text in batches via
// callback. Use this for processing arbitrarily large token sequences without
// buffer limits.
//
// |tokenizer| is the tokenizer to use.
// |ids| is the array of token IDs.
// |id_count| is the number of IDs.
// |flags| controls decoding behavior (e.g., SKIP_SPECIAL_TOKENS).
// |callback| is invoked with batched text strings.
// |user_data| is passed to the callback.
//
// Out-of-range token IDs (negative or >= vocab_size) are silently skipped.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - Status from callback if it returns non-OK
//   - IREE_STATUS_INVALID_ARGUMENT for invalid inputs
IREE_API_EXPORT iree_status_t iree_tokenizer_decode_streaming(
    const iree_tokenizer_t* tokenizer, const int32_t* ids,
    iree_host_size_t id_count, iree_tokenizer_decode_flags_t flags,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

// Decodes token IDs back to text (buffer-based wrapper).
//
// This is a convenience wrapper around iree_tokenizer_decode_streaming that
// collects text into a caller-provided buffer.
//
// |tokenizer| is the tokenizer to use.
// |ids| is the array of token IDs.
// |id_count| is the number of IDs.
// |flags| controls decoding behavior (e.g., SKIP_SPECIAL_TOKENS).
// |out_text| receives the decoded text (caller-allocated).
// |max_text| is the capacity of out_text (including null terminator).
// |out_length| receives the actual text length (not including null).
//
// Out-of-range token IDs (negative or >= vocab_size) are silently skipped.
// This allows robust decoding of partially valid sequences.
//
// Returns:
//   - IREE_STATUS_OK on success (including when some IDs were skipped)
//   - IREE_STATUS_RESOURCE_EXHAUSTED if output buffer too small
//   - IREE_STATUS_INVALID_ARGUMENT for NULL pointer arguments
IREE_API_EXPORT iree_status_t iree_tokenizer_decode(
    const iree_tokenizer_t* tokenizer, const int32_t* ids,
    iree_host_size_t id_count, iree_tokenizer_decode_flags_t flags,
    char* out_text, iree_host_size_t max_text, iree_host_size_t* out_length);

//===----------------------------------------------------------------------===//
// Streaming Encode API (chunk-based)
//===----------------------------------------------------------------------===//

// Maximum literal lookahead buffer size (bytes).
// Covers all known HuggingFace tokenizer added_tokens (e.g.,
// <|begin_of_text|>).
#define IREE_TOKENIZER_LITERAL_LOOKAHEAD_CAPACITY 64

// State for streaming encode across multiple input chunks.
//
// This enables processing arbitrarily large inputs with constant memory by
// feeding chunks incrementally. The state tracks:
// - Incomplete UTF-8 sequences at chunk boundaries
// - Partial literal matches that may span chunks
// - Transform segments that span chunk boundaries
// - Accumulated tokens awaiting callback
//
// Size: ~8.5KB, designed for stack allocation. For constrained stacks, heap
// allocation is also supported.
//
// Usage:
//   iree_tokenizer_encode_stream_state_t state;
//   iree_tokenizer_encode_stream_initialize(&state, tokenizer, flags);
//   while (more_chunks) {
//     iree_tokenizer_encode_stream_feed(&state, chunk, callback, user_data);
//   }
//   iree_tokenizer_encode_stream_finalize(&state, callback, user_data);
//
// Thread Safety:
//   Each state instance must be used by only one thread at a time.
//   Multiple threads can use separate states with the same tokenizer.
typedef struct iree_tokenizer_encode_stream_state_t {
  // Core references.
  const iree_tokenizer_t* tokenizer;    // Reference to tokenizer (not owned).
  iree_tokenizer_encode_flags_t flags;  // Encode flags.

  // Incomplete UTF-8 sequence at chunk boundary.
  // Maximum 3 bytes buffered (4-byte sequence split after first byte means
  // we have at most 3 bytes waiting for completion).
  uint8_t utf8_partial[4];
  uint8_t utf8_partial_length;

  // Literals (added_tokens) can span chunk boundaries. We buffer up to the
  // length of the longest literal to detect matches that cross boundaries.
  char literal_lookahead[IREE_TOKENIZER_LITERAL_LOOKAHEAD_CAPACITY];
  uint8_t literal_lookahead_length;
  uint8_t literal_max_length;  // Cached max literal length from tokenizer.
  // Bitmask of bytes that can start a literal (for fast prefix rejection).
  // Bit at position b is set if any literal starts with byte value b.
  // This allows O(1) rejection of suffixes that can't possibly match.
  uint8_t literal_first_byte_mask[32];

  // Split transform regex execution state (embedded, ~48 bytes).
  // Used only when transform type is SPLIT.
  iree_tokenizer_regex_exec_state_t split_regex_state;
  bool split_regex_initialized;

  // Transform segment carryover buffer.
  // When a transform produces a partial segment at chunk boundary (e.g.,
  // Metaspace split mode mid-word), we buffer it for the next chunk.
  char transform_carryover[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_host_size_t transform_carryover_length;

  // Have we emitted the BOS token yet? (For ADD_SPECIAL_TOKENS mode.)
  bool bos_emitted;
  // Have we started processing text? (To know when to emit BOS.)
  bool text_started;

  // Batched token output buffer (same capacity as non-streaming).
  int32_t token_buffer[IREE_TOKENIZER_TOKEN_BATCH_CAPACITY];
  iree_host_size_t token_count;

  // Total bytes processed so far (for debugging/progress tracking).
  iree_host_size_t total_bytes_processed;
} iree_tokenizer_encode_stream_state_t;

// Initializes streaming encode state.
//
// |state| is caller-owned storage for execution state (stack or heap).
// |tokenizer| is the tokenizer to use (must remain valid during streaming).
// |flags| controls special token handling (ADD_SPECIAL_TOKENS).
//         Note: truncation flags are ignored in streaming mode.
//
// After init, the state is ready to receive chunks via feed().
IREE_API_EXPORT void iree_tokenizer_encode_stream_initialize(
    iree_tokenizer_encode_stream_state_t* state,
    const iree_tokenizer_t* tokenizer, iree_tokenizer_encode_flags_t flags);

// Feeds a chunk of text to the streaming encoder.
//
// Processes the chunk through the encode pipeline, emitting token batches via
// callback as they become ready. The callback may be invoked zero or more times
// per feed() call.
//
// |state| is the streaming state (from init).
// |chunk| is the text chunk to process (UTF-8, need not be complete sequences).
// |callback| is invoked with batched token IDs (may be NULL to discard).
// |user_data| is passed to the callback.
//
// Chunks can be any size and need not align to UTF-8 or word boundaries - the
// state handles partial sequences across chunk boundaries.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - Status from callback if it returns non-OK
//   - IREE_STATUS_INVALID_ARGUMENT for NULL state
IREE_API_EXPORT iree_status_t iree_tokenizer_encode_stream_feed(
    iree_tokenizer_encode_stream_state_t* state, iree_string_view_t chunk,
    iree_tokenizer_token_callback_fn_t callback, void* user_data);

// Finalizes streaming encode, flushing all pending state.
//
// Must be called after all chunks have been fed to:
// 1. Process any buffered partial UTF-8 sequence (invalid → U+FFFD)
// 2. Flush literal lookahead buffer as non-literal text
// 3. Emit any pending transform segment
// 4. Emit EOS token if ADD_SPECIAL_TOKENS and postprocessor template has one
// 5. Flush any remaining accumulated tokens
//
// After finalize, the state should not be reused without initializing again.
//
// |state| is the streaming state.
// |callback| is invoked with final token batches (may be NULL).
// |user_data| is passed to the callback.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - Status from callback if it returns non-OK
IREE_API_EXPORT iree_status_t iree_tokenizer_encode_stream_finalize(
    iree_tokenizer_encode_stream_state_t* state,
    iree_tokenizer_token_callback_fn_t callback, void* user_data);

//===----------------------------------------------------------------------===//
// Truncation Utility
//===----------------------------------------------------------------------===//

// Truncates a token array in-place.
//
// This is a utility for post-processing token arrays to fit model context
// limits. Truncation can be from the left (keeping end) or right (keeping
// start, the default).
//
// |ids| is the array of token IDs to truncate (modified in-place).
// |count| is the current number of tokens in |ids|.
// |max_length| is the maximum number of tokens to keep.
// |flags| controls truncation direction
// (IREE_TOKENIZER_ENCODE_FLAG_TRUNCATE_LEFT). |out_count| receives the final
// token count (may be < count if truncated).
//
// If count <= max_length, no truncation occurs and out_count = count.
IREE_API_EXPORT void iree_tokenizer_truncate(
    int32_t* ids, iree_host_size_t count, iree_host_size_t max_length,
    iree_tokenizer_encode_flags_t flags, iree_host_size_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TOKENIZER_H_
