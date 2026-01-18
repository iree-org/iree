// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Token decoders for converting tokens back to text.
//
// Decoders reverse the encoding process, transforming token strings back into
// human-readable text. Common decoders include:
//
// - WordPiece: Strip "##" prefix from continuation tokens, join without spaces
// - Metaspace: Replace ▁ (U+2581) with spaces
// - ByteLevel: Reverse GPT-2 byte-to-Unicode mapping
//
// Usage (streaming with callback):
//   iree_tokenizer_decoder_t dec;
//   iree_tokenizer_decoder_initialize_wordpiece(&dec);
//   iree_tokenizer_decoder_state_t state;
//   iree_tokenizer_decoder_begin(&dec, &state);
//   iree_tokenizer_decoder_decode(&dec, &state, tokens, callback, user_data);
//   // ... more batches ...
//   iree_tokenizer_decoder_deinitialize(&dec);

#ifndef IREE_TOKENIZER_DECODER_H_
#define IREE_TOKENIZER_DECODER_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Decoder Types
//===----------------------------------------------------------------------===//

// Decoder algorithm type.
typedef enum iree_tokenizer_decoder_type_e {
  // No decoding (passthrough - concatenate tokens).
  IREE_TOKENIZER_DECODER_NONE = 0,
  // WordPiece: strip "##" prefix, join without spaces.
  IREE_TOKENIZER_DECODER_WORDPIECE,
  // Metaspace: replace ▁ with space.
  IREE_TOKENIZER_DECODER_METASPACE,
  // ByteLevel: reverse GPT-2 byte encoding.
  IREE_TOKENIZER_DECODER_BYTE_LEVEL,
  // BPE: basic concatenation (same as NONE for most cases).
  IREE_TOKENIZER_DECODER_BPE,
  // Sequence: chain multiple decoders in order.
  IREE_TOKENIZER_DECODER_SEQUENCE,
  // Replace: generic pattern replacement (pattern → content).
  IREE_TOKENIZER_DECODER_REPLACE,
  // ByteFallback: convert <0xHH> byte tokens to bytes with UTF-8 fallback.
  IREE_TOKENIZER_DECODER_BYTE_FALLBACK,
  // Fuse: concatenate all tokens into a single output string.
  IREE_TOKENIZER_DECODER_FUSE,
  // Strip: remove N chars from token start/end.
  IREE_TOKENIZER_DECODER_STRIP,
} iree_tokenizer_decoder_type_t;

//===----------------------------------------------------------------------===//
// Decoder Configuration
//===----------------------------------------------------------------------===//

// Configuration flags for WordPiece decoder.
typedef enum iree_tokenizer_wordpiece_decoder_flag_bits_e {
  IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_DEFAULT = 0,
  // Clean up tokenization spaces (remove spaces before punctuation).
  IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_CLEANUP_SPACES = 1 << 0,
} iree_tokenizer_wordpiece_decoder_flag_bits_t;
typedef uint32_t iree_tokenizer_wordpiece_decoder_flags_t;

// Configuration for WordPiece decoder.
typedef struct iree_tokenizer_wordpiece_decoder_config_t {
  // Prefix to strip from continuation tokens (default: "##").
  iree_string_view_t prefix;
  // Decoder behavior flags.
  iree_tokenizer_wordpiece_decoder_flags_t flags;
} iree_tokenizer_wordpiece_decoder_config_t;

// Configuration flags for Metaspace decoder.
typedef enum iree_tokenizer_metaspace_decoder_flag_bits_e {
  IREE_TOKENIZER_METASPACE_DECODER_FLAG_DEFAULT = 0,
  // Strip leading replacement from first token (add_prefix_space: true).
  // When set, the decoder removes the leading metaspace that was added during
  // encoding. When not set, leading metaspace is converted to space like any
  // other metaspace.
  IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING = 1 << 0,
} iree_tokenizer_metaspace_decoder_flag_bits_t;
typedef uint32_t iree_tokenizer_metaspace_decoder_flags_t;

// Configuration for Metaspace decoder.
typedef struct iree_tokenizer_metaspace_decoder_config_t {
  // Unicode codepoint that represents space (default: U+2581 ▁).
  uint32_t replacement;
  // Decoder behavior flags.
  iree_tokenizer_metaspace_decoder_flags_t flags;
} iree_tokenizer_metaspace_decoder_config_t;

// Configuration flags for ByteLevel decoder.
typedef enum iree_tokenizer_byte_level_decoder_flag_bits_e {
  IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT = 0,
  // Strip leading space that was added during encode (add_prefix_space).
  IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_ADD_PREFIX_SPACE = 1 << 0,
} iree_tokenizer_byte_level_decoder_flag_bits_t;
typedef uint32_t iree_tokenizer_byte_level_decoder_flags_t;

// Configuration for ByteLevel decoder.
typedef struct iree_tokenizer_byte_level_decoder_config_t {
  // Decoder behavior flags.
  iree_tokenizer_byte_level_decoder_flags_t flags;
} iree_tokenizer_byte_level_decoder_config_t;

// Maximum sizes for inline pattern/content storage in Replace decoder.
#define IREE_TOKENIZER_REPLACE_DECODER_PATTERN_MAX_SIZE 16
#define IREE_TOKENIZER_REPLACE_DECODER_CONTENT_MAX_SIZE 8

// Configuration for Replace decoder.
// Replaces all occurrences of |pattern| with |content| in each token.
// Note: Unlike Metaspace, Replace does NOT strip leading replacement on first
// token. Use Strip decoder after Replace if that behavior is needed.
typedef struct iree_tokenizer_replace_decoder_config_t {
  // Pattern to find and replace (UTF-8 string, stored inline).
  char pattern[IREE_TOKENIZER_REPLACE_DECODER_PATTERN_MAX_SIZE];
  uint8_t pattern_length;
  // Replacement content (UTF-8 string, stored inline).
  char content[IREE_TOKENIZER_REPLACE_DECODER_CONTENT_MAX_SIZE];
  uint8_t content_length;
} iree_tokenizer_replace_decoder_config_t;

// Configuration for ByteFallback decoder.
// No configuration needed - behavior is fixed.
typedef struct iree_tokenizer_byte_fallback_decoder_config_t {
  uint8_t reserved;  // Placeholder for future flags.
} iree_tokenizer_byte_fallback_decoder_config_t;

// Configuration for Fuse decoder.
// No configuration needed - joins all tokens into one string.
typedef struct iree_tokenizer_fuse_decoder_config_t {
  uint8_t reserved;  // Placeholder for future flags.
} iree_tokenizer_fuse_decoder_config_t;

// Maximum size for inline content storage in Strip decoder.
#define IREE_TOKENIZER_STRIP_DECODER_CONTENT_MAX_SIZE 4

// Configuration for Strip decoder.
// Strips up to |start| instances of |content| from token start,
// and up to |stop| instances from token end.
typedef struct iree_tokenizer_strip_decoder_config_t {
  // Character to strip (UTF-8, stored inline, typically 1 byte like space).
  char content[IREE_TOKENIZER_STRIP_DECODER_CONTENT_MAX_SIZE];
  uint8_t content_length;
  // Max characters to strip from start (0 = none).
  uint8_t start;
  // Max characters to strip from end (0 = none).
  uint8_t stop;
} iree_tokenizer_strip_decoder_config_t;

// Forward declaration for self-reference in Sequence config.
typedef struct iree_tokenizer_decoder_t iree_tokenizer_decoder_t;

// Configuration for Sequence decoder (chain of decoders).
// Requires heap allocation for child decoders.
typedef struct iree_tokenizer_sequence_decoder_config_t {
  // Number of child decoders.
  iree_host_size_t count;
  // Heap-allocated array of child decoders (owned).
  iree_tokenizer_decoder_t* children;
  // Allocator used for children array.
  iree_allocator_t allocator;
} iree_tokenizer_sequence_decoder_config_t;

// Decoder instance with inline configuration.
// Most decoders are value types with no heap allocation.
// Sequence decoder owns heap-allocated children and must be deinitialized.
struct iree_tokenizer_decoder_t {
  iree_tokenizer_decoder_type_t type;
  union {
    iree_tokenizer_wordpiece_decoder_config_t wordpiece;
    iree_tokenizer_metaspace_decoder_config_t metaspace;
    iree_tokenizer_byte_level_decoder_config_t byte_level;
    iree_tokenizer_replace_decoder_config_t replace;
    iree_tokenizer_byte_fallback_decoder_config_t byte_fallback;
    iree_tokenizer_fuse_decoder_config_t fuse;
    iree_tokenizer_strip_decoder_config_t strip;
    iree_tokenizer_sequence_decoder_config_t sequence;
  } config;
};

//===----------------------------------------------------------------------===//
// Decoder Lifecycle
//===----------------------------------------------------------------------===//

// Deinitializes a decoder, freeing any owned resources.
// Most decoders are value types with no heap allocation.
// Sequence decoder owns heap-allocated children that are recursively freed.
// MUST be called for all decoders regardless of type to ensure proper cleanup.
void iree_tokenizer_decoder_deinitialize(iree_tokenizer_decoder_t* decoder);

//===----------------------------------------------------------------------===//
// Decoder Initializers
//===----------------------------------------------------------------------===//

// Initializes a decoder that concatenates tokens without modification.
void iree_tokenizer_decoder_initialize_none(
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a WordPiece decoder.
// |prefix| is the continuation prefix to strip. Pass a string_view with
// .data=NULL to use the default "##". Pass an empty string_view with
// .data pointing to "" to disable prefix stripping entirely.
// |flags| controls decoder behavior (e.g., CLEANUP_SPACES).
void iree_tokenizer_decoder_initialize_wordpiece(
    iree_string_view_t prefix, iree_tokenizer_wordpiece_decoder_flags_t flags,
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a Metaspace decoder.
// |replacement| is the codepoint representing space (0 for default ▁).
// |flags| controls decoder behavior (e.g., STRIP_LEADING to remove leading
// metaspace from first token when add_prefix_space was used during encode).
void iree_tokenizer_decoder_initialize_metaspace(
    uint32_t replacement, iree_tokenizer_metaspace_decoder_flags_t flags,
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a ByteLevel decoder.
// |flags| controls decoder behavior (e.g., ADD_PREFIX_SPACE to strip leading
// space that was added during encode).
void iree_tokenizer_decoder_initialize_byte_level(
    iree_tokenizer_byte_level_decoder_flags_t flags,
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a BPE decoder (basic concatenation).
void iree_tokenizer_decoder_initialize_bpe(
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a Replace decoder.
// |pattern| is the UTF-8 string to find (e.g., "▁").
// |content| is the UTF-8 replacement string (e.g., " ").
// Pattern and content are copied inline (limited by struct field sizes).
//
// Returns IREE_STATUS_OK on success.
// Returns IREE_STATUS_INVALID_ARGUMENT if pattern or content exceed capacity.
iree_status_t iree_tokenizer_decoder_initialize_replace(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a ByteFallback decoder.
// Converts tokens like <0x61> to bytes, with UTF-8 validation fallback.
// Invalid UTF-8 sequences are replaced with U+FFFD per invalid byte.
void iree_tokenizer_decoder_initialize_byte_fallback(
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a Fuse decoder.
// Concatenates all input tokens into a single output string.
void iree_tokenizer_decoder_initialize_fuse(
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a Strip decoder.
// |content| is the UTF-8 character to strip (typically " ").
// |start| is max characters to strip from token start.
// |stop| is max characters to strip from token end.
//
// Returns IREE_STATUS_OK on success.
// Returns IREE_STATUS_INVALID_ARGUMENT if content exceeds capacity.
iree_status_t iree_tokenizer_decoder_initialize_strip(
    iree_string_view_t content, uint8_t start, uint8_t stop,
    iree_tokenizer_decoder_t* out_decoder);

// Initializes a Sequence decoder that chains multiple decoders.
// TAKES OWNERSHIP of |children| array (move semantics). The children array and
// its contents are moved to the decoder. Caller must not use or deinitialize
// children after this call.
//
// Returns IREE_STATUS_OK on success.
// Returns IREE_STATUS_INVALID_ARGUMENT if children is NULL and count > 0.
iree_status_t iree_tokenizer_decoder_initialize_sequence(
    iree_tokenizer_decoder_t* children, iree_host_size_t count,
    iree_allocator_t allocator, iree_tokenizer_decoder_t* out_decoder);

//===----------------------------------------------------------------------===//
// Decoder State (for multi-batch streaming)
//===----------------------------------------------------------------------===//

// State for streaming decode across multiple batches.
typedef struct iree_tokenizer_decoder_state_t {
  bool is_first_token;
} iree_tokenizer_decoder_state_t;

// Initializes decoder state for a new decode session.
static inline void iree_tokenizer_decoder_begin(
    const iree_tokenizer_decoder_t* decoder,
    iree_tokenizer_decoder_state_t* out_state) {
  (void)decoder;
  out_state->is_first_token = true;
}

//===----------------------------------------------------------------------===//
// Decoder API
//===----------------------------------------------------------------------===//

// Decodes a batch of tokens, emitting text via callback.
// Symmetric with iree_tokenizer_text_transform_encode.
//
// |decoder| is the decoder configuration.
// |state| tracks decode state across batches (call begin() first).
// |tokens| is a batch of token strings (from vocab lookup).
// |callback| receives decoded text batches.
// |user_data| is passed to the callback.
//
// Returns IREE_STATUS_OK on success, or the first error from callback.
iree_status_t iree_tokenizer_decoder_decode(
    const iree_tokenizer_decoder_t* decoder,
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_DECODER_H_
