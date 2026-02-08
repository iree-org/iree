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

// Maximum bytes required to encode any Unicode codepoint in UTF-8.
// Codepoints U+10000..U+10FFFF require 4 bytes: 11110xxx 10xxxxxx 10xxxxxx
// 10xxxxxx
#define IREE_UNICODE_UTF8_MAX_BYTE_LENGTH 4

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

// Returns the expected UTF-8 sequence length from a lead byte (1-4).
// For invalid lead bytes (continuation bytes 0x80-0xBF), returns 1.
// This is useful for advancing through UTF-8 text byte-by-byte when
// character boundaries are needed but full decoding is not required.
static inline iree_host_size_t iree_unicode_utf8_sequence_length(uint8_t byte) {
  if ((byte & 0x80) == 0) return 1;     // ASCII: 0xxxxxxx
  if ((byte & 0xE0) == 0xC0) return 2;  // 2-byte: 110xxxxx
  if ((byte & 0xF0) == 0xE0) return 3;  // 3-byte: 1110xxxx
  if ((byte & 0xF8) == 0xF0) return 4;  // 4-byte: 11110xxx
  return 1;  // Invalid/continuation byte: advance by 1
}

// Validates that |bytes| forms a valid UTF-8 sequence of |length| bytes.
// Returns true if valid, false if invalid (bad continuation bytes or overlong).
// Faster than utf8_decode() when the codepoint value is not needed.
static inline bool iree_unicode_utf8_is_valid_sequence(
    const uint8_t* bytes, iree_host_size_t length) {
  if (length == 0 || length > 4) return false;

  // Single byte (ASCII): valid if high bit is clear.
  if (length == 1) return (bytes[0] & 0x80) == 0;

  // Check continuation bytes are 10xxxxxx.
  for (iree_host_size_t i = 1; i < length; ++i) {
    if ((bytes[i] & 0xC0) != 0x80) return false;
  }

  // Check not overlong (minimal encoding) and valid codepoint range.
  if (length == 2) {
    // 2-byte: codepoint must be >= 0x80 (first payload nibble non-zero).
    return (bytes[0] & 0x1E) != 0;
  }
  if (length == 3) {
    // 3-byte: codepoint >= 0x800, and not surrogate (0xD800-0xDFFF).
    uint32_t codepoint = ((uint32_t)(bytes[0] & 0x0F) << 12) |
                         ((uint32_t)(bytes[1] & 0x3F) << 6) |
                         (uint32_t)(bytes[2] & 0x3F);
    return codepoint >= 0x800 && (codepoint < 0xD800 || codepoint > 0xDFFF);
  }
  // 4-byte: codepoint >= 0x10000 and <= 0x10FFFF.
  uint32_t codepoint = ((uint32_t)(bytes[0] & 0x07) << 18) |
                       ((uint32_t)(bytes[1] & 0x3F) << 12) |
                       ((uint32_t)(bytes[2] & 0x3F) << 6) |
                       (uint32_t)(bytes[3] & 0x3F);
  return codepoint >= 0x10000 && codepoint <= IREE_UNICODE_MAX_CODEPOINT;
}

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

// General category flags.
// Bits 0-6 encode the major category (first letter of Unicode
// General_Category). Bit 7 encodes the Mn (Nonspacing Mark) subcategory for
// BERT accent stripping.
typedef enum iree_unicode_category_e {
  IREE_UNICODE_CATEGORY_NONE = 0,
  IREE_UNICODE_CATEGORY_LETTER = 1 << 0,       // L (Lu, Ll, Lt, Lm, Lo)
  IREE_UNICODE_CATEGORY_MARK = 1 << 1,         // M (Mn, Mc, Me)
  IREE_UNICODE_CATEGORY_NUMBER = 1 << 2,       // N (Nd, Nl, No)
  IREE_UNICODE_CATEGORY_PUNCTUATION = 1 << 3,  // P (Pc, Pd, Ps, Pe, Pi, Pf, Po)
  IREE_UNICODE_CATEGORY_SYMBOL = 1 << 4,       // S (Sm, Sc, Sk, So)
  IREE_UNICODE_CATEGORY_SEPARATOR = 1 << 5,    // Z (Zs, Zl, Zp)
  IREE_UNICODE_CATEGORY_OTHER = 1 << 6,        // C (Cc, Cf, Cs, Co, Cn)
  // Subcategory flag: set for Mn (Mark, Nonspacing) only.
  // Combined with MARK, so Mn codepoints have both MARK and MARK_NONSPACING
  // set.
  IREE_UNICODE_CATEGORY_MARK_NONSPACING = 1 << 7,  // Mn only
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
// DEPRECATED: Legacy unified table, kept for the 4 entries with both mappings.
typedef struct iree_unicode_case_mapping_t {
  uint32_t codepoint;
  uint32_t lowercase;
  uint32_t uppercase;
} iree_unicode_case_mapping_t;

// Direction-specific case mapping entry (codepoint -> target).
// Used by the split uppercase_mappings and lowercase_mappings tables.
// Smaller struct (8 bytes vs 12 bytes) improves cache efficiency.
typedef struct iree_unicode_case_mapping_simple_t {
  uint32_t codepoint;
  uint32_t target;
} iree_unicode_case_mapping_simple_t;

// Singleton flag for NFD decomposition entries. Set in the high bit of the
// base field for 1:1 decompositions (like CJK Compatibility Ideographs).
// For NFC normalization, only singleton decompositions are applied.
#define IREE_UNICODE_NFD_SINGLETON_FLAG 0x80000000u
#define IREE_UNICODE_NFD_BASE_MASK 0x7FFFFFFFu

// NFD decomposition entry (codepoint -> base + combining mark).
// The base field's high bit indicates singleton (see above); use the mask
// to extract the actual base codepoint.
// The combining field is 0 for singleton decompositions or the combining mark
// codepoint for 2-codepoint decompositions.
typedef struct iree_unicode_nfd_mapping_t {
  uint32_t codepoint;
  uint32_t base;
  uint32_t combining;  // 0 if singleton, else the combining mark.
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

// NFC canonical decomposition entry for NFC_QC=No characters.
// Contains the fully-expanded (recursive) canonical decomposition.
// Unused target slots are zero. Sorted by codepoint for binary search.
typedef struct iree_unicode_nfc_decomp_t {
  uint32_t codepoint;
  uint32_t target[3];  // Decomposed codepoints (0-terminated).
} iree_unicode_nfc_decomp_t;

// Maximum length of an NFKD decomposition.
// The longest is U+FDFA (Arabic ligature) which decomposes to 18 codepoints.
#define IREE_UNICODE_NFKD_MAX_DECOMPOSITION_LENGTH 18

// NFKD compatibility decomposition entry.
// Sorted by codepoint for binary search.
//
// Short decompositions (length <= 4) are stored inline in |inline_targets|.
// Long decompositions use |offset| to index into the overflow array.
typedef struct iree_unicode_nfkd_mapping_t {
  uint32_t codepoint;
  uint16_t length;             // Number of codepoints in decomposition (1-18).
  uint16_t offset;             // For length > 4: index into overflow array.
  uint32_t inline_targets[4];  // Inline storage for short decompositions.
} iree_unicode_nfkd_mapping_t;

// Returns the general category of |codepoint|.
iree_unicode_category_t iree_unicode_category(uint32_t codepoint);

// Returns true if |codepoint| is a letter (L category).
static inline bool iree_unicode_is_letter(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) & IREE_UNICODE_CATEGORY_LETTER) != 0;
}

// Returns true if |codepoint| is a combining mark (M category: Mn, Mc, Me).
static inline bool iree_unicode_is_mark(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) & IREE_UNICODE_CATEGORY_MARK) != 0;
}

// Returns true if |codepoint| is a nonspacing mark (Mn category only).
// This is specifically the Mn subcategory, NOT Mc (Spacing Combining) or
// Me (Enclosing). Used for BERT accent stripping where only Mn should be
// stripped (HuggingFace uses is_mark_nonspacing() from unicode_categories).
static inline bool iree_unicode_is_mark_nonspacing(uint32_t codepoint) {
  return (iree_unicode_category(codepoint) &
          IREE_UNICODE_CATEGORY_MARK_NONSPACING) != 0;
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

// Returns true if |codepoint| is a zero-width or invisible format character
// that should be stripped during BERT-style text cleaning.
//
// These are Unicode Format (Cf) category characters that are invisible and
// used for text layout/formatting rather than content. HuggingFace's BERT
// tokenizer strips these along with control characters when clean_text=true.
//
// Includes:
//   - U+200B-U+200F: Zero-width space, joiners, directional marks
//   - U+202A-U+202E: Directional formatting (LRE, RLE, PDF, LRO, RLO)
//   - U+2060-U+2064: Word joiner, invisible operators
//   - U+FEFF: Byte Order Mark (also zero-width no-break space)
//
// Does NOT include:
//   - U+00A0 (NBSP) - visible whitespace
//   - U+3000 (Ideographic Space) - visible CJK whitespace
//   - Combining marks - handled separately by accent stripping
bool iree_unicode_is_invisible_format(uint32_t codepoint);

// Returns true if |codepoint| is a Han (Chinese) character.
// This includes CJK Unified Ideographs and related blocks used for Chinese,
// Japanese Kanji, and Korean Hanja. Does NOT include Hiragana, Katakana, or
// Hangul (use the specific predicates for those scripts).
bool iree_unicode_is_han(uint32_t codepoint);

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

// Returns the number of lowercase codepoints written to |out| (1 or 2).
// |out| must have space for at least 2 codepoints.
//
// For U+0130 (İ, Latin Capital Letter I With Dot Above), returns 2 codepoints:
// [U+0069, U+0307] (i + combining dot above). This is the ONLY unconditional
// 1:N lowercase mapping in Unicode (per SpecialCasing.txt).
//
// For all other codepoints, returns 1 codepoint (the lowercase equivalent,
// or the input unchanged if it has no lowercase mapping).
iree_host_size_t iree_unicode_to_lower(uint32_t codepoint, uint32_t out[2]);

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
// For full decomposition (1:N mappings), use iree_unicode_decompose().
uint32_t iree_unicode_nfd_base(uint32_t codepoint);

// Full canonical (NFD) decomposition of |codepoint| into constituent
// codepoints. Writes decomposed codepoints to |out_codepoints| and returns the
// count. |out_codepoints| must have space for at least 4 codepoints.
//
// Handles:
// - Hangul syllables (U+AC00-U+D7A3): algorithmic decomposition to 2-3 Jamo
// - Precomposed Latin/Greek/Cyrillic: table lookup returning base + combining
// - Characters without decomposition: returns the input unchanged (count=1)
//
// Used for NFD normalization and accent stripping. For accent stripping,
// decompose then filter out combining marks (codepoints with CCC > 0).
//
// Example: '뮨' (U+BBA8) -> [U+1106, U+1172, U+11AB] (count=3)
// Example: 'é' (U+00E9) -> [U+0065, U+0301] (count=2, base + combining acute)
// Example: 'A' (U+0041) -> [U+0041] (count=1, unchanged)
iree_host_size_t iree_unicode_decompose(uint32_t codepoint,
                                        uint32_t* out_codepoints);

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
//   |capacity| must be >= input.size (output can only shrink via composition).
//   Uses a small fixed internal buffer (~128 bytes) for processing.
//
// Combining sequence limit:
//   Returns IREE_STATUS_RESOURCE_EXHAUSTED if any combining sequence exceeds
//   32 codepoints (starter + combining marks). This limit exceeds Unicode's
//   Stream-Safe Text Format (30 characters) and handles all real-world text.
//
// Requires valid UTF-8 input. Behavior is undefined for invalid sequences.
iree_status_t iree_unicode_compose(iree_string_view_t input, char* out_buffer,
                                   iree_host_size_t capacity,
                                   iree_host_size_t* out_length);

// Applies full NFC (Canonical Composition) normalization to UTF-8 input.
// This performs the complete NFC algorithm:
//   1. NFD decomposition (canonically decompose all characters)
//   2. Canonical ordering (sort combining marks by CCC)
//   3. Canonical composition (combine base + combining -> precomposed)
//
// Unlike iree_unicode_compose() which only performs steps 2-3, this function
// handles characters that require decomposition first, such as:
//   - CJK Compatibility Ideographs (U+2F800-U+2FA1D)
//   - Characters with canonical decompositions that differ from their form
//
// Use this function when processing text that may contain non-NFC characters,
// such as tokenizer inputs from arbitrary sources.
//
// Buffer requirements:
//   |out_capacity| should be >= input.size * 4 to handle worst-case NFD
//   expansion (Hangul syllables can decompose to 3 Jamo, each up to 3 bytes
//   UTF-8). In practice, most text does not expand significantly.
//
// Requires valid UTF-8 input. Behavior is undefined for invalid sequences.
iree_status_t iree_unicode_nfc(iree_string_view_t input,
                               iree_host_size_t out_capacity, char* out_buffer,
                               iree_host_size_t* out_length);

// Decomposes only singleton mappings and Hangul syllables.
// Writes decomposed codepoints to |out_codepoints| (must have space for 4).
// Returns the number of codepoints written (1-3).
//
// This is a subset of full decomposition, handling only:
// - Hangul syllables (U+AC00-U+D7A3): algorithmic decomposition to 2-3 Jamo
// - CJK Compatibility Ideographs: singleton decomposition to canonical form
// - All other characters: unchanged (including precomposed like é)
//
// Used by NFC normalization where precomposed characters should be preserved,
// but compatibility mappings and Hangul must still be decomposed for proper
// canonical composition.
//
// Unlike iree_unicode_decompose() which decomposes ALL precomposed characters
// (for accent stripping), this function preserves them (é stays as é).
iree_host_size_t iree_unicode_decompose_singleton(uint32_t codepoint,
                                                  uint32_t* out_codepoints);

// Decomposes a codepoint for NFC canonical normalization.
// Writes decomposed codepoints to |out_codepoints| (must have space for 4).
// Returns the number of codepoints written (1-4).
//
// Handles ALL NFC-required decompositions:
// - Hangul syllables (U+AC00-U+D7A3): algorithmic decomposition to 2-3 Jamo
// - CJK Compatibility Ideographs: singleton decomposition to canonical form
// - NFC_QC=No characters: full recursive canonical decomposition (singletons,
//   Indic nukta combinations, Tibetan subjoined, Hebrew dagesh, Musical
//   symbols)
// - All other characters: unchanged
//
// Unlike iree_unicode_decompose_singleton, this function handles the complete
// set of characters that must be decomposed for NFC correctness, including
// multi-codepoint canonical decompositions and singleton NFC_QC=No characters
// not in the CJK compatibility range.
iree_host_size_t iree_unicode_decompose_nfc_canonical(uint32_t codepoint,
                                                      uint32_t* out_codepoints);

// NFKD (Normalization Form Compatibility Decomposition) of |codepoint|.
// Writes decomposed codepoints to |out_codepoints|.
// |out_codepoints| must have space for
// IREE_UNICODE_NFKD_MAX_DECOMPOSITION_LENGTH codepoints (18). Returns the
// number of codepoints written (1-18).
//
// NFKD = NFD (canonical decomposition) + compatibility decomposition:
// - Canonical (NFD): é → e + combining acute (preserves visual appearance)
// - Compatibility (K): ﬁ → fi, ① → 1, ㎞ → km (normalizes for comparison)
//
// Handles:
// - Compatibility decompositions (table lookup)
// - Hangul syllables (algorithmic to 2-3 Jamo)
// - Recursive canonical decomposition on results
//
// Used by NFKD normalizer for tokenizers like XLNet and ALBERT.
iree_host_size_t iree_unicode_decompose_nfkd(uint32_t codepoint,
                                             uint32_t* out_codepoints);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_UNICODE_H_
