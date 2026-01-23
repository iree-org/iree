// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Text transforms for tokenization encode/decode.
//
// Text transforms convert between raw text and an intermediate representation
// suitable for tokenization. Each transform supports both encode (for input
// processing) and decode (for output generation):
//
// - BERT: Splits on whitespace, isolates punctuation (encode)
//         Concatenates pieces back together (decode)
// - Metaspace: Replaces spaces with ▁ (encode), ▁ back to spaces (decode)
// - ByteLevel: Maps bytes to Unicode chars (encode), chars back to bytes
// (decode)
//
// Usage:
//   static iree_status_t my_callback(void* user_data,
//                                    iree_string_view_list_t segments) {
//     for (iree_host_size_t i = 0; i < segments.count; ++i) {
//       // process segments.values[i]
//     }
//     return iree_ok_status();
//   }
//
//   iree_tokenizer_text_transform_t xform;
//   iree_tokenizer_text_transform_initialize_bert(&xform);
//   iree_tokenizer_text_transform_encode(&xform, text, my_callback, user_data);
//   iree_tokenizer_text_transform_deinitialize(&xform);

#ifndef IREE_TOKENIZER_TRANSFORMS_TRANSFORM_H_
#define IREE_TOKENIZER_TRANSFORMS_TRANSFORM_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/regex/exec.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Text Transform Types
//===----------------------------------------------------------------------===//

// Text transform algorithm type.
typedef enum iree_tokenizer_text_transform_type_e {
  // No transformation (passthrough).
  IREE_TOKENIZER_TEXT_TRANSFORM_NONE = 0,
  // BERT-style: split on whitespace, isolate punctuation.
  IREE_TOKENIZER_TEXT_TRANSFORM_BERT,
  // SentencePiece-style: replace spaces with replacement char (▁).
  IREE_TOKENIZER_TEXT_TRANSFORM_METASPACE,
  // GPT-2 style: map bytes to Unicode characters.
  IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL,
  // Simple whitespace splitting only.
  IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE,
  // Sequence of transforms applied in order.
  IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE,
  // Regex-based splitting (GPT-2/Llama/Qwen style).
  IREE_TOKENIZER_TEXT_TRANSFORM_SPLIT,
} iree_tokenizer_text_transform_type_t;

// Metaspace prepend scheme (when to add the replacement character).
typedef enum iree_tokenizer_prepend_scheme_e {
  // Always prepend to each word.
  IREE_TOKENIZER_PREPEND_ALWAYS = 0,
  // Only prepend to the first word.
  IREE_TOKENIZER_PREPEND_FIRST,
  // Never prepend.
  IREE_TOKENIZER_PREPEND_NEVER,
} iree_tokenizer_prepend_scheme_t;

//===----------------------------------------------------------------------===//
// Text Transform Configuration
//===----------------------------------------------------------------------===//

// Metaspace transform configuration flags.
typedef enum iree_tokenizer_metaspace_flag_bits_e {
  IREE_TOKENIZER_METASPACE_FLAG_DEFAULT = 0,
  // Split output on the replacement character into separate segments.
  IREE_TOKENIZER_METASPACE_FLAG_SPLIT = 1 << 0,
} iree_tokenizer_metaspace_flag_bits_t;
typedef uint32_t iree_tokenizer_metaspace_flags_t;

// Configuration for Metaspace transform.
typedef struct iree_tokenizer_metaspace_config_t {
  // Unicode codepoint to replace whitespace with (default: U+2581 ▁).
  uint32_t replacement;
  // When to prepend the replacement character.
  iree_tokenizer_prepend_scheme_t prepend_scheme;
  // Configuration flags (IREE_TOKENIZER_METASPACE_*).
  iree_tokenizer_metaspace_flags_t flags;
} iree_tokenizer_metaspace_config_t;

// ByteLevel transform configuration flags.
typedef enum iree_tokenizer_byte_level_flag_bits_e {
  IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT = 0,
  // Prepend a space to the input if it doesn't start with ASCII space (0x20).
  IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE = 1 << 0,
} iree_tokenizer_byte_level_flag_bits_t;
typedef uint32_t iree_tokenizer_byte_level_flags_t;

// Configuration for ByteLevel transform.
typedef struct iree_tokenizer_byte_level_config_t {
  // Configuration flags (IREE_TOKENIZER_BYTE_LEVEL_*).
  iree_tokenizer_byte_level_flags_t flags;
} iree_tokenizer_byte_level_config_t;

// Forward declaration for sequence config.
typedef struct iree_tokenizer_text_transform_t iree_tokenizer_text_transform_t;

// Configuration for Sequence transform (chain of transforms).
// Unlike other configs, this requires heap allocation for the children array.
typedef struct iree_tokenizer_sequence_config_t {
  // Number of child transforms.
  iree_host_size_t count;
  // Heap-allocated array of child transforms (owned).
  iree_tokenizer_text_transform_t* children;
  // Allocator used to free children array.
  iree_allocator_t allocator;
} iree_tokenizer_sequence_config_t;

// Configuration for Split transform (regex-based splitting).
// Unlike simple configs, this requires heap allocation for the compiled DFA.
// See iree/tokenizer/regex/exec.h for DFA details.
typedef struct iree_tokenizer_split_config_t {
  // Compiled DFA for pattern matching (view into pattern_storage).
  iree_tokenizer_regex_dfa_t dfa;
  // Heap-allocated DFA binary data (owned).
  uint8_t* pattern_storage;
  // How to handle matched patterns.
  iree_tokenizer_regex_split_behavior_t behavior;
  // Whether to invert the pattern (emit matches instead of gaps).
  bool invert;
  // Allocator used to free pattern_storage.
  iree_allocator_t allocator;
} iree_tokenizer_split_config_t;

// Text transform instance with inline configuration.
// Most transforms are value types (~24 bytes) with no heap allocation.
// Sequence transforms own heap-allocated children and must be deinitialized.
typedef struct iree_tokenizer_text_transform_t {
  iree_tokenizer_text_transform_type_t type;
  union {
    iree_tokenizer_metaspace_config_t metaspace;
    iree_tokenizer_byte_level_config_t byte_level;
    iree_tokenizer_sequence_config_t sequence;
    iree_tokenizer_split_config_t split;
  } config;
  // Optional normalizer applied inline during encode (before segmentation).
  // Set type to IREE_TOKENIZER_NORMALIZER_NONE if no normalization needed.
  iree_tokenizer_normalizer_t normalizer;
} iree_tokenizer_text_transform_t;

// Deinitializes a transform, freeing any owned resources.
// MUST be called for all transforms regardless of type to ensure proper
// cleanup. For Sequence transforms, this frees the children array and
// recursively deinitializes each child. For other transform types, this is a
// no-op but must still be called to allow generic transform handling. The
// transform struct itself is not freed (caller manages its storage).
void iree_tokenizer_text_transform_deinitialize(
    iree_tokenizer_text_transform_t* transform);

//===----------------------------------------------------------------------===//
// Text Transform Initializers
//===----------------------------------------------------------------------===//

// Initializes a transform that passes through text unchanged.
static inline void iree_tokenizer_text_transform_initialize_none(
    iree_tokenizer_text_transform_t* out_transform) {
  memset(out_transform, 0, sizeof(*out_transform));
  out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_NONE;
  iree_tokenizer_normalizer_initialize_none(&out_transform->normalizer);
}

// Initializes a BERT-style transform (whitespace + punctuation splitting).
static inline void iree_tokenizer_text_transform_initialize_bert(
    iree_tokenizer_text_transform_t* out_transform) {
  memset(out_transform, 0, sizeof(*out_transform));
  out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_BERT;
  iree_tokenizer_normalizer_initialize_none(&out_transform->normalizer);
}

// Initializes a Metaspace transform with the given configuration.
// |replacement| is the Unicode codepoint to replace spaces with (0 for default
// ▁). |prepend_scheme| controls when to prepend the replacement. |flags|
// controls additional behavior (IREE_TOKENIZER_METASPACE_*).
static inline void iree_tokenizer_text_transform_initialize_metaspace(
    uint32_t replacement, iree_tokenizer_prepend_scheme_t prepend_scheme,
    iree_tokenizer_metaspace_flags_t flags,
    iree_tokenizer_text_transform_t* out_transform) {
  memset(out_transform, 0, sizeof(*out_transform));
  out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_METASPACE;
  out_transform->config.metaspace.replacement =
      replacement ? replacement : IREE_TOKENIZER_METASPACE_REPLACEMENT;
  out_transform->config.metaspace.prepend_scheme = prepend_scheme;
  out_transform->config.metaspace.flags = flags;
  iree_tokenizer_normalizer_initialize_none(&out_transform->normalizer);
}

// Initializes a ByteLevel transform with the given configuration.
// |flags| controls behavior (IREE_TOKENIZER_BYTE_LEVEL_*).
static inline void iree_tokenizer_text_transform_initialize_byte_level(
    iree_tokenizer_byte_level_flags_t flags,
    iree_tokenizer_text_transform_t* out_transform) {
  memset(out_transform, 0, sizeof(*out_transform));
  out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL;
  out_transform->config.byte_level.flags = flags;
  iree_tokenizer_normalizer_initialize_none(&out_transform->normalizer);
}

// Initializes a simple whitespace-splitting transform.
static inline void iree_tokenizer_text_transform_initialize_whitespace(
    iree_tokenizer_text_transform_t* out_transform) {
  memset(out_transform, 0, sizeof(*out_transform));
  out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE;
  iree_tokenizer_normalizer_initialize_none(&out_transform->normalizer);
}

// Initializes a Sequence transform that chains multiple transforms.
//
// WARNING: TAKES OWNERSHIP of |children| via move semantics. The children array
// is copied into heap-allocated storage and the originals are zeroed to prevent
// double-free. After this call, the source transforms become NONE (passthrough)
// and should not be used for their original purpose.
//
// It is safe (but unnecessary) to call deinitialize on moved-from children.
// Using a moved-from transform will behave as NONE (passthrough), not as the
// original transform type.
//
// On encode, transforms are applied in order (0, 1, 2, ...).
// On decode, transforms are applied in reverse order (..., 2, 1, 0).
iree_status_t iree_tokenizer_text_transform_initialize_sequence(
    iree_tokenizer_text_transform_t* children, iree_host_size_t count,
    iree_allocator_t allocator, iree_tokenizer_text_transform_t* out_transform);

// Initializes a Split transform with a regex pattern.
//
// Compiles the regex pattern to a DFA for fast matching. The pattern uses
// syntax compatible with GPT-2/Llama/Qwen tokenizers (see regex/compile.h).
//
// |pattern| is the regex pattern to compile.
// |behavior| determines how matched patterns are handled (see exec.h):
//   IREE_TOKENIZER_REGEX_SPLIT_REMOVED: discard matched portions
//   IREE_TOKENIZER_REGEX_SPLIT_ISOLATED: emit matched portions as segments
//   IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_PREVIOUS: append to previous segment
//   IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT: prepend to next segment
//   IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS: merge consecutive matches
// |invert| if true, emits matched portions instead of gaps between them.
// |allocator| is used to allocate the compiled DFA storage.
// |out_transform| receives the initialized transform.
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - IREE_STATUS_INVALID_ARGUMENT if the pattern has syntax errors or
//     behavior is out of range.
//   - IREE_STATUS_RESOURCE_EXHAUSTED if DFA is too large.
iree_status_t iree_tokenizer_text_transform_initialize_split(
    iree_string_view_t pattern, iree_tokenizer_regex_split_behavior_t behavior,
    bool invert, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform);

//===----------------------------------------------------------------------===//
// Text Transform Encode API
//===----------------------------------------------------------------------===//

// Encodes text using the specified transform, emitting segments via callback.
//
// Transforms the input text into segments suitable for tokenization.
// For BERT/Whitespace: splits into word segments (zero-copy into input).
// For Metaspace: replaces whitespace with ▁, optionally splits.
// For ByteLevel: maps each byte to a Unicode character.
//
// Segments are emitted in batches via the callback. Each batch is valid only
// for the duration of the callback invocation. No dynamic allocation is
// performed; transforms use stack buffers for batching.
//
// |transform| is the transform configuration.
// |text| is the input text to encode.
// |callback| receives batches of output segments.
// |user_data| is passed to the callback.
//
// Returns IREE_STATUS_OK on success, or the first error from callback.
//
// |normalizer| is the normalizer to apply during encoding. If NULL, uses
// the transform's embedded normalizer. Sequence transforms pass their
// normalizer to children via this parameter.
iree_status_t iree_tokenizer_text_transform_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_text_transform_t* transform, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data);

//===----------------------------------------------------------------------===//
// Text Transform Decode API
//===----------------------------------------------------------------------===//

// Decodes text using the specified transform.
//
// Reverses the encode transformation to recover the original text.
// For BERT/Whitespace: passes through unchanged (tokenizer handles joining).
// For Metaspace: replaces ▁ back to spaces.
// For ByteLevel: maps Unicode characters back to bytes.
//
// The output is written directly to the provided buffer. Inverse transforms
// typically produce output that is equal or smaller than input, so the same
// buffer used for input can safely be reused as the output buffer.
//
// |transform| is the transform configuration.
// |text| is the encoded text to decode.
// |out_buffer| receives the decoded text (caller-provided, may overlap |text|).
// |max_size| is the capacity of out_buffer.
// |out_size| receives the actual decoded size (not including null terminator).
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_RESOURCE_EXHAUSTED if output would exceed max_size
iree_status_t iree_tokenizer_text_transform_decode(
    const iree_tokenizer_text_transform_t* transform, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TRANSFORMS_TRANSFORM_H_
