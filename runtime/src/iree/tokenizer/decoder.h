// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Decoder interface for token-to-text conversion.
//
// Decoders transform token strings (from vocab lookup) back to text.
// Examples: ByteFallback (byte tokens), Metaspace (▁→space), WordPiece (##).
//
// The decode pipeline is:
//   Token IDs → Vocab Lookup → Token Strings → Decoder → Final Text
//
// Design principles:
// - Pull-based: caller controls flow via output buffer size
// - Batched: processes array of token strings, not one at a time
// - Zero allocation: state lives in caller-provided storage
// - Composable: Sequence decoder chains multiple decoders

#ifndef IREE_TOKENIZER_DECODER_H_
#define IREE_TOKENIZER_DECODER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_tokenizer_decoder_t iree_tokenizer_decoder_t;
typedef struct iree_tokenizer_decoder_state_t iree_tokenizer_decoder_state_t;
typedef struct iree_tokenizer_decoder_vtable_t iree_tokenizer_decoder_vtable_t;

//===----------------------------------------------------------------------===//
// Token String List (for batched input)
//===----------------------------------------------------------------------===//

// List of token strings for batched decoder input.
// Each string is a token's text representation from vocab lookup.
typedef struct iree_tokenizer_string_list_t {
  iree_host_size_t count;
  const iree_string_view_t* values;
} iree_tokenizer_string_list_t;

// Creates a string list from an array and count.
static inline iree_tokenizer_string_list_t iree_tokenizer_make_string_list(
    const iree_string_view_t* values, iree_host_size_t count) {
  iree_tokenizer_string_list_t list = {count, values};
  return list;
}

// Returns an empty string list.
static inline iree_tokenizer_string_list_t iree_tokenizer_string_list_empty(
    void) {
  iree_tokenizer_string_list_t list = {0, NULL};
  return list;
}

//===----------------------------------------------------------------------===//
// Decoder Flags
//===----------------------------------------------------------------------===//

// Flags declaring decoder transformation properties.
// Used by Sequence decoder to validate children at construction time.
typedef uint32_t iree_tokenizer_decoder_flags_t;
enum iree_tokenizer_decoder_flags_e {
  IREE_TOKENIZER_DECODER_FLAG_NONE = 0,
  // Output byte count is always <= input byte count.
  // Enables zero-allocation in-place transformation in Sequence decoder.
  IREE_TOKENIZER_DECODER_FLAG_SHRINKING = 1u << 0,
};

//===----------------------------------------------------------------------===//
// Decoder Capabilities (for pre-decode optimization)
//===----------------------------------------------------------------------===//

// Capabilities declaring decoder context-dependency.
// Used by tokenizer to determine if decoded token strings can be pre-computed
// at build time, eliminating runtime decoder work during decode.
typedef uint32_t iree_tokenizer_decoder_capability_t;
enum iree_tokenizer_decoder_capability_e {
  IREE_TOKENIZER_DECODER_CAPABILITY_NONE = 0,
  // Output for each token depends only on that token's text, not on adjacent
  // tokens or stream position. Enables full pre-decode at build time.
  // Examples: ByteLevel, Replace, Strip.
  IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS = 1u << 0,
  // Output for the first token in a stream differs from subsequent tokens.
  // Typically the first token omits a leading space that "rest" tokens include.
  // Requires STATELESS to also be set for pre-decode to work.
  // Examples: Metaspace (add_prefix_space), WordPiece.
  IREE_TOKENIZER_DECODER_CAPABILITY_POSITION_SENSITIVE = 1u << 1,
  // Stateless for all tokens EXCEPT byte tokens (<0xHH> pattern).
  // Byte tokens require cross-token UTF-8 accumulation (stateful), but
  // non-byte tokens are pure passthrough. Enables a hybrid pre-decode path:
  // non-byte tokens are pre-decoded at build time, byte tokens are handled
  // with an inline accumulator at decode time.
  // Examples: ByteFallback.
  IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS_EXCEPT_BYTE_TOKENS = 1u << 2,
};

//===----------------------------------------------------------------------===//
// Decoder Base Type
//===----------------------------------------------------------------------===//

// Base decoder structure. All concrete decoder types embed this at
// offset 0. The vtable provides type-specific operations; state_size is
// cached at creation time for efficient state allocation.
struct iree_tokenizer_decoder_t {
  const iree_tokenizer_decoder_vtable_t* vtable;  // Must be at offset 0.
  // Size of state struct, cached at creation.
  iree_host_size_t state_size;
  iree_tokenizer_decoder_capability_t capabilities;  // Pre-decode properties.
};

// Base streaming state structure. All concrete state types embed this at
// offset 0. The decoder back-pointer enables vtable dispatch from state.
struct iree_tokenizer_decoder_state_t {
  const iree_tokenizer_decoder_t* decoder;
};

//===----------------------------------------------------------------------===//
// Decoder VTable
//===----------------------------------------------------------------------===//

// VTable for decoder implementations.
//
// Pull-based processing model:
// - state_process() consumes token strings and writes text output
// - Caller controls batch size via input string count and output buffer
// - Incomplete sequences (e.g., partial byte fallback) buffered in state
// - state_finalize() flushes remaining buffered data
struct iree_tokenizer_decoder_vtable_t {
  void (*destroy)(iree_tokenizer_decoder_t* decoder);
  iree_status_t (*state_initialize)(const iree_tokenizer_decoder_t* decoder,
                                    void* storage,
                                    iree_tokenizer_decoder_state_t** out_state);
  void (*state_deinitialize)(iree_tokenizer_decoder_state_t* state);
  iree_status_t (*state_process)(iree_tokenizer_decoder_state_t* state,
                                 iree_tokenizer_string_list_t token_strings,
                                 iree_mutable_string_view_t output,
                                 iree_host_size_t* out_strings_consumed,
                                 iree_host_size_t* out_bytes_written);
  iree_status_t (*state_finalize)(iree_tokenizer_decoder_state_t* state,
                                  iree_mutable_string_view_t output,
                                  iree_host_size_t* out_written);
  bool (*state_has_pending)(const iree_tokenizer_decoder_state_t* state);
  iree_tokenizer_decoder_flags_t flags;
};

//===----------------------------------------------------------------------===//
// Decoder Public API
//===----------------------------------------------------------------------===//

// Initializes base decoder fields. Called by concrete implementations
// after allocating the decoder struct.
void iree_tokenizer_decoder_initialize(
    iree_tokenizer_decoder_t* decoder,
    const iree_tokenizer_decoder_vtable_t* vtable, iree_host_size_t state_size,
    iree_tokenizer_decoder_capability_t capabilities);

// Frees a decoder. Safe to call with NULL.
void iree_tokenizer_decoder_free(iree_tokenizer_decoder_t* decoder);

// Returns the size of state storage required for this decoder.
static inline iree_host_size_t iree_tokenizer_decoder_state_size(
    const iree_tokenizer_decoder_t* decoder) {
  return decoder->state_size;
}

// Returns the capability flags for this decoder.
static inline iree_tokenizer_decoder_capability_t
iree_tokenizer_decoder_capabilities(const iree_tokenizer_decoder_t* decoder) {
  return decoder->capabilities;
}

// Initializes streaming state in caller-provided storage.
// |storage| must be at least iree_tokenizer_decoder_state_size() bytes.
static inline iree_status_t iree_tokenizer_decoder_state_initialize(
    const iree_tokenizer_decoder_t* decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  return decoder->vtable->state_initialize(decoder, storage, out_state);
}

// Deinitializes streaming state (does not free storage).
static inline void iree_tokenizer_decoder_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  if (state && state->decoder) {
    state->decoder->vtable->state_deinitialize(state);
  }
}

// Minimum output buffer size for decoders. Smaller buffers cannot guarantee
// progress on multi-byte sequences (UTF-8 max 4 bytes, replacement char 3).
#define IREE_TOKENIZER_DECODER_MIN_BUFFER_SIZE 4

// Processes a batch of token strings, writing decoded text.
//
// Reads up to |token_strings.count| strings and writes up to |output.size|
// bytes. Returns strings consumed in |out_strings_consumed| and bytes written
// in |out_bytes_written|.
//
// A token string is "consumed" (counted in |out_strings_consumed|) only if it
// has been fully processed or successfully buffered into internal state. If a
// token cannot be accepted (e.g., output buffer full and no internal buffering
// available), it is not counted. The caller should retry with the unconsumed
// tokens after draining output.
//
// May consume fewer strings than count if the output buffer is full or the
// decoder is waiting for more tokens to complete a sequence (e.g., byte
// fallback). May write fewer bytes than output capacity if input is exhausted
// or the decoder needs more tokens to produce output.
static inline iree_status_t iree_tokenizer_decoder_state_process(
    iree_tokenizer_decoder_state_t* state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  IREE_ASSERT(output.size >= IREE_TOKENIZER_DECODER_MIN_BUFFER_SIZE);
  return state->decoder->vtable->state_process(
      state, token_strings, output, out_strings_consumed, out_bytes_written);
}

// Finalizes the stream, flushing all buffered data. Call after all tokens
// have been provided via state_process(). May need multiple calls if the
// output buffer is too small for buffered data.
static inline iree_status_t iree_tokenizer_decoder_state_finalize(
    iree_tokenizer_decoder_state_t* state, iree_mutable_string_view_t output,
    iree_host_size_t* out_written) {
  return state->decoder->vtable->state_finalize(state, output, out_written);
}

// Returns true if state has pending data that would produce output on
// finalize. Useful for checking if more finalize() calls are needed.
static inline bool iree_tokenizer_decoder_state_has_pending(
    const iree_tokenizer_decoder_state_t* state) {
  return state->decoder->vtable->state_has_pending(state);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_H_
