// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Text normalizers for tokenization preprocessing.
//
// Normalizers transform input text before tokenization to ensure consistent
// handling of case, accents, whitespace, and other text properties. Common
// normalizers include:
//
// - BertNormalizer: Lowercase, strip accents, clean control chars
// - Lowercase: Convert to lowercase
// - StripAccents: Remove diacritical marks (é→e, ñ→n)
// - Sequence: Chain multiple normalizers
//
// Usage:
//   iree_tokenizer_normalizer_t norm;
//   iree_tokenizer_normalizer_initialize_bert(true, true, &norm);
//   iree_tokenizer_normalizer_apply(&norm, input, output, capacity, &length);
//   iree_tokenizer_normalizer_deinitialize(&norm);

#ifndef IREE_TOKENIZER_NORMALIZER_H_
#define IREE_TOKENIZER_NORMALIZER_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/regex/exec.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Normalizer Types
//===----------------------------------------------------------------------===//

// Normalizer algorithm type.
typedef enum iree_tokenizer_normalizer_type_e {
  // No normalization (passthrough).
  IREE_TOKENIZER_NORMALIZER_NONE = 0,
  // BERT-style: lowercase, strip accents, clean control chars.
  IREE_TOKENIZER_NORMALIZER_BERT,
  // Simple lowercase transformation.
  IREE_TOKENIZER_NORMALIZER_LOWERCASE,
  // Strip accents/diacritical marks.
  IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS,
  // Sequence of normalizers applied in order.
  IREE_TOKENIZER_NORMALIZER_SEQUENCE,
  // Prepend a string to the input.
  IREE_TOKENIZER_NORMALIZER_PREPEND,
  // Replace pattern with content (literal string match).
  IREE_TOKENIZER_NORMALIZER_REPLACE,
  // Replace pattern with content (regex match).
  IREE_TOKENIZER_NORMALIZER_REPLACE_REGEX,
  // NFC (Canonical Composition) normalization.
  IREE_TOKENIZER_NORMALIZER_NFC,
  // SentencePiece precompiled character map normalizer.
  IREE_TOKENIZER_NORMALIZER_PRECOMPILED,
  // Strip whitespace from left/right of input.
  IREE_TOKENIZER_NORMALIZER_STRIP,
} iree_tokenizer_normalizer_type_t;

//===----------------------------------------------------------------------===//
// Normalizer Configuration
//===----------------------------------------------------------------------===//

// Configuration flags for BertNormalizer.
typedef enum iree_tokenizer_bert_normalizer_flag_bits_e {
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT = 0,
  // Clean text by removing control characters.
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT = 1 << 0,
  // Add spaces around CJK characters.
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS = 1 << 1,
  // Strip accents from characters (é→e).
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS = 1 << 2,
  // Convert text to lowercase.
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE = 1 << 3,
} iree_tokenizer_bert_normalizer_flag_bits_t;
typedef uint32_t iree_tokenizer_bert_normalizer_flags_t;

// Configuration for BertNormalizer.
typedef struct iree_tokenizer_bert_normalizer_config_t {
  iree_tokenizer_bert_normalizer_flags_t flags;
} iree_tokenizer_bert_normalizer_config_t;

// Forward declaration for sequence config.
typedef struct iree_tokenizer_normalizer_t iree_tokenizer_normalizer_t;

// Configuration for Sequence normalizer (chain of normalizers).
typedef struct iree_tokenizer_normalizer_sequence_config_t {
  // Number of child normalizers.
  iree_host_size_t count;
  // Heap-allocated array of child normalizers (owned).
  iree_tokenizer_normalizer_t* children;
  // Allocator used to free children array.
  iree_allocator_t allocator;
} iree_tokenizer_normalizer_sequence_config_t;

// Maximum size for inline prepend string storage.
#define IREE_TOKENIZER_PREPEND_NORMALIZER_MAX_SIZE 16

// Configuration for Prepend normalizer.
// Stores the prepend string inline.
typedef struct iree_tokenizer_prepend_normalizer_config_t {
  char prepend[IREE_TOKENIZER_PREPEND_NORMALIZER_MAX_SIZE];
  uint8_t length;  // Length of prepend string in bytes.
} iree_tokenizer_prepend_normalizer_config_t;

// Maximum sizes for inline pattern/content storage in Replace normalizer.
#define IREE_TOKENIZER_REPLACE_NORMALIZER_PATTERN_MAX_SIZE 8
#define IREE_TOKENIZER_REPLACE_NORMALIZER_CONTENT_MAX_SIZE 8

// Configuration for Replace normalizer (literal string match).
// Stores pattern and content inline.
typedef struct iree_tokenizer_replace_normalizer_config_t {
  char pattern[IREE_TOKENIZER_REPLACE_NORMALIZER_PATTERN_MAX_SIZE];
  char content[IREE_TOKENIZER_REPLACE_NORMALIZER_CONTENT_MAX_SIZE];
  uint8_t pattern_length;  // Length of pattern in bytes.
  uint8_t content_length;  // Length of content in bytes.
} iree_tokenizer_replace_normalizer_config_t;

// Configuration for Replace-Regex normalizer (regex pattern match).
// Uses the regex engine for pattern matching. Stores compiled DFA on heap.
typedef struct iree_tokenizer_normalizer_replace_regex_config_t {
  uint8_t* dfa_data;               // Compiled DFA binary (heap-allocated).
  iree_host_size_t dfa_size;       // Size of DFA data in bytes.
  iree_tokenizer_regex_dfa_t dfa;  // Loaded DFA handle.
  char content[32];                // Replacement content (inline).
  uint8_t content_length;          // Length of content in bytes.
  iree_allocator_t allocator;      // For cleanup.
} iree_tokenizer_normalizer_replace_regex_config_t;

// Configuration for SentencePiece Precompiled normalizer.
// Uses a double-array trie for efficient prefix matching of Unicode character
// normalization rules compiled from SentencePiece models.
typedef struct iree_tokenizer_precompiled_normalizer_config_t {
  // Heap-allocated double-array trie (owned).
  // Each element is a 32-bit unit encoding offset, label, and leaf flag.
  uint32_t* trie;
  // Number of elements in the trie array.
  iree_host_size_t trie_count;
  // Heap-allocated normalized string containing replacement strings (owned).
  // Entries are null-terminated and concatenated.
  char* normalized;
  // Total length of the normalized string in bytes.
  iree_host_size_t normalized_length;
  // Allocator used to free trie and normalized.
  iree_allocator_t allocator;
} iree_tokenizer_precompiled_normalizer_config_t;

// Configuration for Strip normalizer.
// Strips whitespace from input text.
typedef struct iree_tokenizer_strip_normalizer_config_t {
  bool strip_left;   // Strip whitespace from beginning.
  bool strip_right;  // Strip whitespace from end.
} iree_tokenizer_strip_normalizer_config_t;

// Normalizer instance with inline configuration.
// Most normalizers are value types with no heap allocation.
// Sequence normalizers own heap-allocated children and must be deinitialized.
typedef struct iree_tokenizer_normalizer_t {
  iree_tokenizer_normalizer_type_t type;
  union {
    iree_tokenizer_bert_normalizer_config_t bert;
    iree_tokenizer_normalizer_sequence_config_t sequence;
    iree_tokenizer_prepend_normalizer_config_t prepend;
    iree_tokenizer_replace_normalizer_config_t replace;
    iree_tokenizer_normalizer_replace_regex_config_t replace_regex;
    iree_tokenizer_precompiled_normalizer_config_t precompiled;
    iree_tokenizer_strip_normalizer_config_t strip;
  } config;
} iree_tokenizer_normalizer_t;

//===----------------------------------------------------------------------===//
// Normalizer Lifecycle
//===----------------------------------------------------------------------===//

// Deinitializes a normalizer, freeing any owned resources.
// MUST be called for all normalizers regardless of type.
void iree_tokenizer_normalizer_deinitialize(
    iree_tokenizer_normalizer_t* normalizer);

//===----------------------------------------------------------------------===//
// Normalizer Initializers
//===----------------------------------------------------------------------===//

// Initializes a normalizer that passes through text unchanged.
void iree_tokenizer_normalizer_initialize_none(
    iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a BERT-style normalizer.
// |flags| controls which normalizations to apply (IREE_TOKENIZER_BERT_*).
void iree_tokenizer_normalizer_initialize_bert(
    iree_tokenizer_bert_normalizer_flags_t flags,
    iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a lowercase normalizer.
void iree_tokenizer_normalizer_initialize_lowercase(
    iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a strip-accents normalizer.
void iree_tokenizer_normalizer_initialize_strip_accents(
    iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a Sequence normalizer that chains multiple normalizers.
// TAKES OWNERSHIP of |children| (move semantics). The children array is copied
// into heap storage and the originals are zeroed. Caller must not deinitialize
// children after this call.
iree_status_t iree_tokenizer_normalizer_initialize_sequence(
    iree_tokenizer_normalizer_t* children, iree_host_size_t count,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a Prepend normalizer that adds a prefix to input text.
// |prepend| is the string to prepend.
iree_status_t iree_tokenizer_normalizer_initialize_prepend(
    iree_string_view_t prepend, iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a Replace normalizer that substitutes pattern with content.
// |pattern| is the string to find.
// |content| is the replacement string.
iree_status_t iree_tokenizer_normalizer_initialize_replace(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a Replace-Regex normalizer from a pre-compiled DFA.
// TAKES OWNERSHIP of |dfa_data| - caller must not free it.
// |dfa_data| is the compiled regex DFA binary.
// |dfa_size| is the size of the DFA data in bytes.
// |content| is the replacement string (max 31 bytes).
// |allocator| is used to free dfa_data during deinitialization.
iree_status_t iree_tokenizer_normalizer_initialize_replace_regex(
    uint8_t* dfa_data, iree_host_size_t dfa_size, iree_string_view_t content,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t* out_normalizer);

// Initializes an NFC (Canonical Composition) normalizer.
// Converts decomposed characters to precomposed form (e.g., e + ́ → é).
void iree_tokenizer_normalizer_initialize_nfc(
    iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a Precompiled normalizer from raw binary data.
// This is the SentencePiece precompiled_charsmap format containing a
// double-array trie and normalized string. TAKES OWNERSHIP of a copy of the
// data - caller retains ownership of the input buffer.
//
// |data| is the raw binary precompiled_charsmap (trie_size + trie +
// normalized). |data_length| is the total length of the data in bytes.
// |allocator| is used to allocate internal storage.
iree_status_t iree_tokenizer_normalizer_initialize_precompiled(
    const uint8_t* data, iree_host_size_t data_length,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t* out_normalizer);

// Initializes a Strip normalizer that removes whitespace from input edges.
// |strip_left| removes whitespace from the beginning.
// |strip_right| removes whitespace from the end.
void iree_tokenizer_normalizer_initialize_strip(
    bool strip_left, bool strip_right,
    iree_tokenizer_normalizer_t* out_normalizer);

//===----------------------------------------------------------------------===//
// Normalizer Apply API
//===----------------------------------------------------------------------===//

// Applies the normalizer to input text, writing result to output buffer.
//
// |normalizer| is the normalizer configuration.
// |input| is the input text to normalize.
// |out_buffer| receives the normalized text (caller-provided).
// |capacity| is the capacity of out_buffer in bytes.
// |out_length| receives the actual normalized length (not including null).
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_RESOURCE_EXHAUSTED if output would exceed capacity
//
// The output buffer should have capacity for up to 3x the input size in the
// worst case (when handle_chinese_chars adds spaces around every character).
// For typical text, output size equals or is less than input size.
iree_status_t iree_tokenizer_normalizer_apply(
    const iree_tokenizer_normalizer_t* normalizer, iree_string_view_t input,
    char* out_buffer, iree_host_size_t capacity, iree_host_size_t* out_length);

// Normalizes a single codepoint, returning normalized codepoints.
//
// This is the streaming-compatible API for inline normalization in
// pre-tokenizers. No allocation is performed.
//
// |normalizer| is the normalizer configuration.
// |codepoint| is the input codepoint to normalize.
// |out_codepoints| receives the normalized codepoints (caller-provided, size
// 4). |out_count| receives the number of normalized codepoints (0-4).
//
// A count of 0 means the codepoint should be skipped (e.g., control character
// when clean_text is enabled, or combining mark when strip_accents is enabled).
void iree_tokenizer_normalizer_normalize_codepoint(
    const iree_tokenizer_normalizer_t* normalizer, uint32_t codepoint,
    uint32_t* out_codepoints, iree_host_size_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_H_
