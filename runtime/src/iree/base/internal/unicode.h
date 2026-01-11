// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unicode utilities for text processing.
//
// Provides UTF-8 encoding/decoding, Unicode category classification, case
// folding, and simple NFD decomposition. Designed for tokenizer preprocessing
// (BERT BasicTokenizer, etc.) with minimal footprint (~50KB tables).

#ifndef IREE_BASE_INTERNAL_UNICODE_H_
#define IREE_BASE_INTERNAL_UNICODE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Unicode replacement character (U+FFFD), returned for invalid sequences.
#define IREE_UNICODE_REPLACEMENT_CHAR 0xFFFDu

// Maximum codepoint value (U+10FFFF).
#define IREE_UNICODE_MAX_CODEPOINT 0x10FFFFu

//===----------------------------------------------------------------------===//
// UTF-8 Codec
//===----------------------------------------------------------------------===//

// Decodes a single UTF-8 codepoint from |text| starting at |*position|.
// Advances |*position| past the decoded sequence. Returns
// IREE_UNICODE_REPLACEMENT_CHAR for invalid sequences (and advances by 1 byte).
//
// Precondition: |*position| < |text.size|. If called at end-of-input,
// returns IREE_UNICODE_REPLACEMENT_CHAR without advancing (callers must guard).
//
// Example:
//   iree_host_size_t position = 0;
//   while (position < text.size) {
//     uint32_t codepoint = iree_unicode_utf8_decode(text, &position);
//     // process codepoint...
//   }
uint32_t iree_unicode_utf8_decode(iree_string_view_t text,
                                  iree_host_size_t* position);

// Encodes a single codepoint to UTF-8, writing to |out_buffer|.
// Returns the number of bytes written (1-4), or 0 if |codepoint| is invalid.
// |out_buffer| must have space for at least 4 bytes.
int iree_unicode_utf8_encode(uint32_t codepoint, char* out_buffer);

// Returns the number of bytes needed to encode |codepoint| in UTF-8 (1-4),
// or 0 if |codepoint| is invalid.
int iree_unicode_utf8_encoded_length(uint32_t codepoint);

// Returns the number of codepoints in a UTF-8 string.
// Each invalid byte is counted as one codepoint (replacement character).
iree_host_size_t iree_unicode_utf8_codepoint_count(iree_string_view_t text);

// Validates that |text| contains only valid UTF-8 sequences.
// Returns true if valid, false if any invalid sequences are found.
bool iree_unicode_utf8_validate(iree_string_view_t text);

// Returns the number of trailing bytes that form an incomplete UTF-8 sequence.
// Used to detect multi-byte UTF-8 characters split across buffer boundaries.
//
// When processing UTF-8 in fixed-size buffers, the last few bytes may be the
// start of a multi-byte sequence whose continuation bytes haven't arrived yet.
// This function detects such incomplete tails so callers can:
// 1. Flush only complete bytes
// 2. Carry over incomplete bytes to the next buffer
//
// Returns 0 if the buffer ends with complete sequences (or is empty).
// Returns 1-3 if the last 1-3 bytes form an incomplete sequence.
//
// Example: Buffer ending with [0xC3] returns 1 (start of 2-byte sequence).
// Example: Buffer ending with [0xE2, 0x80] returns 2 (incomplete 3-byte seq).
iree_host_size_t iree_unicode_utf8_incomplete_tail_length(
    const char* data, iree_host_size_t size);

//===----------------------------------------------------------------------===//
// Unicode Categories
//===----------------------------------------------------------------------===//

// General category flags (major categories only).
// These correspond to the first letter of Unicode General_Category values.
typedef enum iree_unicode_category_e {
  IREE_UNICODE_CATEGORY_NONE = 0,
  IREE_UNICODE_CATEGORY_LETTER = 1 << 0,       // L (Lu, Ll, Lt, Lm, Lo)
  IREE_UNICODE_CATEGORY_MARK = 1 << 1,         // M (Mn, Mc, Me)
  IREE_UNICODE_CATEGORY_NUMBER = 1 << 2,       // N (Nd, Nl, No)
  IREE_UNICODE_CATEGORY_PUNCTUATION = 1 << 3,  // P (Pc, Pd, Ps, Pe, Pi, Pf, Po)
  IREE_UNICODE_CATEGORY_SYMBOL = 1 << 4,       // S (Sm, Sc, Sk, So)
  IREE_UNICODE_CATEGORY_SEPARATOR = 1 << 5,    // Z (Zs, Zl, Zp)
  IREE_UNICODE_CATEGORY_OTHER = 1 << 6,        // C (Cc, Cf, Cs, Co, Cn)
} iree_unicode_category_t;

//===----------------------------------------------------------------------===//
// Internal Table Structures
//===----------------------------------------------------------------------===//

// Category range entry for binary search lookup.
typedef struct iree_unicode_category_range_t {
  uint32_t start;
  uint32_t end;
  uint8_t category;
} iree_unicode_category_range_t;

// Case mapping entry (codepoint -> lowercase/uppercase).
typedef struct iree_unicode_case_mapping_t {
  uint32_t codepoint;
  uint32_t lowercase;
  uint32_t uppercase;
} iree_unicode_case_mapping_t;

// NFD decomposition entry (codepoint -> base character).
typedef struct iree_unicode_nfd_mapping_t {
  uint32_t codepoint;
  uint32_t base;
} iree_unicode_nfd_mapping_t;

// Canonical Combining Class (CCC) entry for combining marks.
typedef struct iree_unicode_ccc_entry_t {
  uint32_t codepoint;
  uint8_t ccc;
} iree_unicode_ccc_entry_t;

// NFC composition pair: (base, combining) -> composed.
typedef struct iree_unicode_nfc_pair_t {
  uint32_t base;
  uint32_t combining;
  uint32_t composed;
} iree_unicode_nfc_pair_t;

// Returns the general category of |codepoint|.
iree_unicode_category_t iree_unicode_category(uint32_t codepoint);

// Returns true if |codepoint| is a letter (L category).
static inline bool iree_unicode_is_letter(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) & IREE_UNICODE_CATEGORY_LETTER) != 0;
}

// Returns true if |codepoint| is a combining mark (M category).
static inline bool iree_unicode_is_mark(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) & IREE_UNICODE_CATEGORY_MARK) != 0;
}

// Returns true if |codepoint| is a number (N category).
static inline bool iree_unicode_is_number(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) & IREE_UNICODE_CATEGORY_NUMBER) != 0;
}

// Returns true if |codepoint| is punctuation (P category).
static inline bool iree_unicode_is_punctuation(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) &
          IREE_UNICODE_CATEGORY_PUNCTUATION) != 0;
}

// Returns true if |codepoint| is a symbol (S category).
static inline bool iree_unicode_is_symbol(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) & IREE_UNICODE_CATEGORY_SYMBOL) != 0;
}

// Returns true if |codepoint| is a separator (Z category).
static inline bool iree_unicode_is_separator(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) & IREE_UNICODE_CATEGORY_SEPARATOR) !=
         0;
}

// Returns true if |codepoint| is in the Other category (C - control, format,
// surrogate, private use, unassigned).
static inline bool iree_unicode_is_other(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) & IREE_UNICODE_CATEGORY_OTHER) != 0;
}

// Returns true if |codepoint| has the White_Space property.
// This is a derived property that includes more than just the Z category.
bool iree_unicode_is_whitespace(uint32_t codepoint);

// Returns true if |codepoint| is a control character (Cc category).
bool iree_unicode_is_control(uint32_t codepoint);

// Returns true if |codepoint| is a CJK character.
// This includes CJK Unified Ideographs and related blocks (Han characters).
bool iree_unicode_is_cjk(uint32_t codepoint);

// Returns true if |codepoint| is Hiragana (U+3040-U+309F).
bool iree_unicode_is_hiragana(uint32_t codepoint);

// Returns true if |codepoint| is Katakana (U+30A0-U+30FF, U+31F0-U+31FF).
bool iree_unicode_is_katakana(uint32_t codepoint);

// Returns true if |codepoint| is Hangul (Korean).
// Includes Hangul Syllables, Jamo, and Compatibility Jamo blocks.
bool iree_unicode_is_hangul(uint32_t codepoint);

//===----------------------------------------------------------------------===//
// Case Folding
//===----------------------------------------------------------------------===//

// Returns the lowercase equivalent of |codepoint|, or |codepoint| unchanged
// if it has no lowercase mapping.
uint32_t iree_unicode_to_lower(uint32_t codepoint);

// Returns the uppercase equivalent of |codepoint|, or |codepoint| unchanged
// if it has no uppercase mapping.
uint32_t iree_unicode_to_upper(uint32_t codepoint);

//===----------------------------------------------------------------------===//
// Simple NFD Decomposition
//===----------------------------------------------------------------------===//

// Returns the NFD base character of |codepoint| (strips combining marks).
// For precomposed characters like 'é' (U+00E9), returns 'e' (U+0065).
// For characters without decomposition, returns |codepoint| unchanged.
//
// This is a simplified 1:1 mapping suitable for BERT-style accent stripping.
// It handles common Latin, Greek, and Cyrillic accented characters.
// For full NFD normalization (1:N mappings), use a dedicated Unicode library.
uint32_t iree_unicode_nfd_base(uint32_t codepoint);

//===----------------------------------------------------------------------===//
// Unicode Composition
//===----------------------------------------------------------------------===//

// Returns the Canonical Combining Class (CCC) for |codepoint|.
// Returns 0 for base characters (starters), non-zero for combining marks.
// CCC values are used to canonically order combining marks.
uint8_t iree_unicode_ccc(uint32_t codepoint);

// Looks up a canonical composition: base + combining -> composed.
// Returns the composed codepoint, or 0 if no composition exists.
// This is a building block for NFC normalization.
uint32_t iree_unicode_compose_pair(uint32_t base, uint32_t combining);

// Applies canonical ordering and composition to UTF-8 input.
// Writes result to |out_buffer| (max |capacity| bytes).
// Returns the output length in |*out_length|.
//
// IMPORTANT: This is NOT full NFC normalization. It performs:
//   1. Canonical ordering (sort combining marks by CCC)
//   2. Canonical composition (combine base + combining -> precomposed)
//
// It does NOT perform NFD decomposition first, which full NFC requires.
// Full NFC is: Compose(Decompose(input)). This function only does Compose().
//
// This is sufficient when:
//   - Input is already in NFD form (fully decomposed)
//   - Input is already in NFC form (nothing to compose)
//   - Input is from typical user keyboards (OS normalizes to NFC)
//
// This may produce incorrect results when:
//   - Input has precomposed characters followed by additional combining marks
//     that would interact differently if the precomposed char were decomposed
//   - Example: é (U+00E9) + combining mark might need decomposition first
//
// For tokenizer use cases (BERT, GPT, LLaMA), this limitation is acceptable:
//   - BERT uses NFD + accent stripping, not NFC
//   - GPT-2/LLaMA use no normalizer at all
//   - Real-world input text is typically already NFC from OS input methods
//
// If full NFC is needed later, add iree_unicode_nfc() that decomposes first.
//
// Buffer requirements:
//   |capacity| must be >= iree_unicode_utf8_codepoint_count(input) * 4
//   The function uses |out_buffer| as scratch space for codepoint processing.
//   For ASCII-only input, only input.size bytes are needed (fast path).
//
// Note: Assumes input is valid UTF-8. Invalid sequences pass through unchanged.
iree_status_t iree_unicode_compose(iree_string_view_t input, char* out_buffer,
                                   iree_host_size_t capacity,
                                   iree_host_size_t* out_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_UNICODE_H_
