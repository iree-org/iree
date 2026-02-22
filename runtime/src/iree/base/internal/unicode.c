// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Table declarations (defined in unicode_tables.c)
//===----------------------------------------------------------------------===//

extern const iree_unicode_category_range_t iree_unicode_category_ranges[];
extern const iree_host_size_t iree_unicode_category_ranges_count;

extern const uint32_t iree_unicode_whitespace_codepoints[];
extern const iree_host_size_t iree_unicode_whitespace_count;

extern const iree_unicode_case_mapping_simple_t
    iree_unicode_lowercase_mappings[];
extern const iree_host_size_t iree_unicode_lowercase_mappings_count;

extern const iree_unicode_case_mapping_simple_t
    iree_unicode_uppercase_mappings[];
extern const iree_host_size_t iree_unicode_uppercase_mappings_count;

extern const iree_unicode_nfd_mapping_t iree_unicode_nfd_mappings[];
extern const iree_host_size_t iree_unicode_nfd_mappings_count;

extern const iree_unicode_ccc_entry_t iree_unicode_ccc_entries[];
extern const iree_host_size_t iree_unicode_ccc_entries_count;

extern const iree_unicode_nfc_pair_t iree_unicode_nfc_pairs[];
extern const iree_host_size_t iree_unicode_nfc_pairs_count;

extern const iree_unicode_nfc_decomp_t iree_unicode_nfc_decompositions[];
extern const iree_host_size_t iree_unicode_nfc_decompositions_count;

extern const iree_unicode_nfkd_mapping_t iree_unicode_nfkd_mappings[];
extern const iree_host_size_t iree_unicode_nfkd_mappings_count;

extern const uint32_t iree_unicode_nfkd_overflow[];
extern const iree_host_size_t iree_unicode_nfkd_overflow_count;

//===----------------------------------------------------------------------===//
// UTF-8 Codec
//===----------------------------------------------------------------------===//

int iree_unicode_utf8_encode(uint32_t codepoint, char* out_buffer) {
  if (codepoint > IREE_UNICODE_MAX_CODEPOINT ||
      (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
    return 0;
  }

  uint8_t* buffer = (uint8_t*)out_buffer;
  if (codepoint < 0x80) {
    buffer[0] = (uint8_t)codepoint;
    return 1;
  } else if (codepoint < 0x800) {
    buffer[0] = (uint8_t)(0xC0 | (codepoint >> 6));
    buffer[1] = (uint8_t)(0x80 | (codepoint & 0x3F));
    return 2;
  } else if (codepoint < 0x10000) {
    buffer[0] = (uint8_t)(0xE0 | (codepoint >> 12));
    buffer[1] = (uint8_t)(0x80 | ((codepoint >> 6) & 0x3F));
    buffer[2] = (uint8_t)(0x80 | (codepoint & 0x3F));
    return 3;
  } else {
    buffer[0] = (uint8_t)(0xF0 | (codepoint >> 18));
    buffer[1] = (uint8_t)(0x80 | ((codepoint >> 12) & 0x3F));
    buffer[2] = (uint8_t)(0x80 | ((codepoint >> 6) & 0x3F));
    buffer[3] = (uint8_t)(0x80 | (codepoint & 0x3F));
    return 4;
  }
}

int iree_unicode_utf8_encoded_length(uint32_t codepoint) {
  if (codepoint > IREE_UNICODE_MAX_CODEPOINT ||
      (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
    return 0;
  }
  if (codepoint < 0x80) return 1;
  if (codepoint < 0x800) return 2;
  if (codepoint < 0x10000) return 3;
  return 4;
}

iree_host_size_t iree_unicode_utf8_codepoint_count(iree_string_view_t text) {
  iree_host_size_t count = 0;
  iree_host_size_t position = 0;
  while (position < text.size) {
    iree_unicode_utf8_decode(text, &position);
    ++count;
  }
  return count;
}

bool iree_unicode_utf8_validate(iree_string_view_t text) {
  iree_host_size_t position = 0;
  while (position < text.size) {
    iree_host_size_t start_position = position;
    uint32_t codepoint = iree_unicode_utf8_decode(text, &position);
    // If we only advanced by 1 byte but got replacement char, it was invalid
    // (unless the input was actually U+FFFD encoded as 3 bytes).
    if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR &&
        position == start_position + 1) {
      // Check if this was actually a valid U+FFFD sequence.
      if (text.size - start_position >= 3) {
        const uint8_t* data = (const uint8_t*)text.data + start_position;
        if (data[0] == 0xEF && data[1] == 0xBF && data[2] == 0xBD) {
          // This was a valid U+FFFD, continue.
          continue;
        }
      }
      return false;
    }
  }
  return true;
}

iree_host_size_t iree_unicode_utf8_incomplete_tail_length(
    const char* data, iree_host_size_t size) {
  if (size == 0) return 0;

  // Scan backwards to find a lead byte (max 3 bytes back since longest
  // sequence is 4 bytes and we need at least 1 byte present).
  iree_host_size_t scan_limit = size < 4 ? size : 4;
  for (iree_host_size_t i = 1; i <= scan_limit; ++i) {
    uint8_t byte = (uint8_t)data[size - i];

    // Continuation bytes (10xxxxxx) are not lead bytes - keep scanning.
    if ((byte & 0xC0) == 0x80) continue;

    // Found a lead byte. Use the sequence length helper.
    iree_host_size_t expected_length = iree_unicode_utf8_sequence_length(byte);

    // Available bytes from lead to end of buffer.
    iree_host_size_t available = i;
    if (available < expected_length) {
      // Incomplete - return number of bytes in the partial sequence.
      return available;
    }
    // Complete sequence - no incomplete tail.
    return 0;
  }

  // Scanned 4 bytes of continuations with no lead byte - malformed, treat as
  // complete to avoid infinite loops.
  return 0;
}

//===----------------------------------------------------------------------===//
// Binary search helpers
//===----------------------------------------------------------------------===//

// Binary search for codepoint in category ranges.
// Returns OTHER for valid but unassigned codepoints (Cn category).
iree_unicode_category_t iree_unicode_category_lookup(uint32_t codepoint) {
  if (iree_unicode_category_ranges_count == 0) {
    return IREE_UNICODE_CATEGORY_OTHER;
  }

  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_category_ranges_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const iree_unicode_category_range_t* range =
        &iree_unicode_category_ranges[mid];
    if (codepoint < range->start) {
      high = mid;
    } else if (codepoint > range->end) {
      low = mid + 1;
    } else {
      return (iree_unicode_category_t)range->category;
    }
  }
  // Codepoint not in any range - treat as unassigned (Cn), which is OTHER.
  return IREE_UNICODE_CATEGORY_OTHER;
}

// Binary search for codepoint in whitespace list.
bool iree_unicode_whitespace_lookup(uint32_t codepoint) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_whitespace_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    uint32_t mid_value = iree_unicode_whitespace_codepoints[mid];
    if (codepoint < mid_value) {
      high = mid;
    } else if (codepoint > mid_value) {
      low = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

// Binary search for codepoint in lowercase mappings.
static const iree_unicode_case_mapping_simple_t* iree_unicode_lowercase_lookup(
    uint32_t codepoint) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_lowercase_mappings_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const iree_unicode_case_mapping_simple_t* mapping =
        &iree_unicode_lowercase_mappings[mid];
    if (codepoint < mapping->codepoint) {
      high = mid;
    } else if (codepoint > mapping->codepoint) {
      low = mid + 1;
    } else {
      return mapping;
    }
  }
  return NULL;
}

// Binary search for codepoint in uppercase mappings.
static const iree_unicode_case_mapping_simple_t* iree_unicode_uppercase_lookup(
    uint32_t codepoint) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_uppercase_mappings_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const iree_unicode_case_mapping_simple_t* mapping =
        &iree_unicode_uppercase_mappings[mid];
    if (codepoint < mapping->codepoint) {
      high = mid;
    } else if (codepoint > mapping->codepoint) {
      low = mid + 1;
    } else {
      return mapping;
    }
  }
  return NULL;
}

// Binary search for codepoint in NFD mappings.
static const iree_unicode_nfd_mapping_t* iree_unicode_nfd_lookup(
    uint32_t codepoint) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_nfd_mappings_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const iree_unicode_nfd_mapping_t* mapping = &iree_unicode_nfd_mappings[mid];
    if (codepoint < mapping->codepoint) {
      high = mid;
    } else if (codepoint > mapping->codepoint) {
      low = mid + 1;
    } else {
      return mapping;
    }
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Unicode Categories
//===----------------------------------------------------------------------===//

// Direct lookup table for Latin-1 Supplement (U+0080-U+00FF).
// Eliminates binary search (~9 comparisons) for the most common non-ASCII
// range. This is the hot path for ByteLevel pre-tokenizers (GPT-2, Llama-3,
// etc.) which map raw bytes 0x80-0xFF to codepoints in this range.
//
// Categories from Unicode 15.0 General_Category property:
//   U+0080-009F: Cc (Control)           → OTHER
//   U+00A0:      Zs (Space Separator)   → SEPARATOR
//   U+00A1-00BF: Mixed Po/Sc/So/Sk/Sm/Pi/Pf/No/Lo/Cf
//   U+00C0-00D6: Lu (Uppercase Letter)  → LETTER
//   U+00D7:      Sm (Math Symbol)       → SYMBOL  (×)
//   U+00D8-00F6: Lu/Ll (Letter)         → LETTER
//   U+00F7:      Sm (Math Symbol)       → SYMBOL  (÷)
//   U+00F8-00FF: Ll (Lowercase Letter)  → LETTER
// clang-format off
const uint8_t iree_unicode_latin1_categories[128] = {
    // U+0080-008F: C1 Controls (Cc).
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
    // U+0090-009F: C1 Controls (Cc).
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
    // U+00A0-00AF:
    //   A0=Zs  A1=Po  A2=Sc  A3=Sc  A4=Sc  A5=Sc  A6=So  A7=Po
    //   A8=Sk  A9=So  AA=Lo  AB=Pi  AC=Sm  AD=Cf  AE=So  AF=Sk
    0x20, 0x08, 0x10, 0x10, 0x10, 0x10, 0x10, 0x08,
    0x10, 0x10, 0x01, 0x08, 0x10, 0x40, 0x10, 0x10,
    // U+00B0-00BF:
    //   B0=So  B1=Sm  B2=No  B3=No  B4=Sk  B5=Ll  B6=So  B7=Po
    //   B8=Sk  B9=No  BA=Lo  BB=Pf  BC=No  BD=No  BE=No  BF=Po
    0x10, 0x10, 0x04, 0x04, 0x10, 0x01, 0x10, 0x08,
    0x10, 0x04, 0x01, 0x08, 0x04, 0x04, 0x04, 0x08,
    // U+00C0-00CF: Lu (Uppercase Letter).
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    // U+00D0-00DF: Lu except D7=Sm (×).
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x10,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    // U+00E0-00EF: Ll (Lowercase Letter).
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    // U+00F0-00FF: Ll except F7=Sm (÷).
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x10,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
};
// clang-format on

bool iree_unicode_is_control(uint32_t codepoint) {
  // Cc category: C0 controls (0x00-0x1F), DEL (0x7F), C1 controls (0x80-0x9F).
  return codepoint <= 0x1F || (codepoint >= 0x7F && codepoint <= 0x9F);
}

bool iree_unicode_is_invisible_format(uint32_t codepoint) {
  // Zero-width and invisible Unicode Format (Cf) characters that should be
  // stripped during BERT-style text cleaning. These characters are used for
  // text layout/formatting but have no visible representation.
  //
  // HuggingFace's BERT tokenizer strips these when clean_text=true.
  switch (codepoint) {
    // Zero-width characters (U+200B-U+200F).
    case 0x200B:  // Zero Width Space (ZWSP)
    case 0x200C:  // Zero Width Non-Joiner (ZWNJ)
    case 0x200D:  // Zero Width Joiner (ZWJ)
    case 0x200E:  // Left-to-Right Mark (LRM)
    case 0x200F:  // Right-to-Left Mark (RLM)
    // Directional formatting (U+202A-U+202E).
    case 0x202A:  // Left-to-Right Embedding (LRE)
    case 0x202B:  // Right-to-Left Embedding (RLE)
    case 0x202C:  // Pop Directional Formatting (PDF)
    case 0x202D:  // Left-to-Right Override (LRO)
    case 0x202E:  // Right-to-Left Override (RLO)
    // Word joiner and invisible operators (U+2060-U+2064).
    case 0x2060:  // Word Joiner (WJ)
    case 0x2061:  // Function Application (invisible)
    case 0x2062:  // Invisible Times
    case 0x2063:  // Invisible Separator
    case 0x2064:  // Invisible Plus
    // Byte Order Mark (U+FEFF).
    case 0xFEFF:  // BOM / Zero Width No-Break Space (ZWNBSP)
      return true;
    default:
      return false;
  }
}

bool iree_unicode_is_han(uint32_t codepoint) {
  // Han (Chinese) characters: CJK Unified Ideographs and related blocks.
  return (codepoint >= 0x4E00 &&
          codepoint <= 0x9FFF) ||  // CJK Unified Ideographs
         (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||    // CJK Extension A
         (codepoint >= 0x20000 && codepoint <= 0x2A6DF) ||  // CJK Extension B
         (codepoint >= 0x2A700 && codepoint <= 0x2B73F) ||  // CJK Extension C
         (codepoint >= 0x2B740 && codepoint <= 0x2B81F) ||  // CJK Extension D
         (codepoint >= 0x2B820 && codepoint <= 0x2CEAF) ||  // CJK Extension E
         (codepoint >= 0xF900 &&
          codepoint <= 0xFAFF) ||  // CJK Compatibility Ideographs
         (codepoint >= 0x2F800 &&
          codepoint <= 0x2FA1F);  // CJK Compat Ideographs Supp
}

bool iree_unicode_is_hiragana(uint32_t codepoint) {
  return codepoint >= 0x3040 && codepoint <= 0x309F;
}

bool iree_unicode_is_katakana(uint32_t codepoint) {
  return (codepoint >= 0x30A0 && codepoint <= 0x30FF) ||  // Katakana
         (codepoint >= 0x31F0 && codepoint <= 0x31FF);  // Katakana Phonetic Ext
}

bool iree_unicode_is_hangul(uint32_t codepoint) {
  return (codepoint >= 0xAC00 && codepoint <= 0xD7AF) ||  // Hangul Syllables
         (codepoint >= 0x1100 && codepoint <= 0x11FF) ||  // Hangul Jamo
         (codepoint >= 0x3130 && codepoint <= 0x318F);    // Hangul Compat Jamo
}

//===----------------------------------------------------------------------===//
// Case Folding
//===----------------------------------------------------------------------===//

iree_host_size_t iree_unicode_to_lower(uint32_t codepoint, uint32_t out[2]) {
  // Fast path for ASCII (most common case).
  if (codepoint >= 'A' && codepoint <= 'Z') {
    out[0] = codepoint + ('a' - 'A');
    return 1;
  }
  if (codepoint < 0x80) {
    out[0] = codepoint;
    return 1;
  }

  // Special case: U+0130 (İ) → i + combining dot (U+0069 U+0307).
  // This is the ONLY unconditional 1:N lowercase mapping in Unicode.
  // See: https://www.unicode.org/Public/UCD/latest/ucd/SpecialCasing.txt
  if (codepoint == 0x0130) {
    out[0] = 0x0069;  // LATIN SMALL LETTER I
    out[1] = 0x0307;  // COMBINING DOT ABOVE
    return 2;
  }

  // Table lookup for non-ASCII.
  const iree_unicode_case_mapping_simple_t* mapping =
      iree_unicode_lowercase_lookup(codepoint);
  if (mapping) {
    out[0] = mapping->target;
    return 1;
  }
  out[0] = codepoint;
  return 1;
}

uint32_t iree_unicode_to_upper(uint32_t codepoint) {
  // Fast path for ASCII.
  if (codepoint >= 'a' && codepoint <= 'z') {
    return codepoint - ('a' - 'A');
  }
  if (codepoint < 0x80) {
    return codepoint;
  }
  const iree_unicode_case_mapping_simple_t* mapping =
      iree_unicode_uppercase_lookup(codepoint);
  if (mapping) {
    return mapping->target;
  }
  return codepoint;
}

//===----------------------------------------------------------------------===//
// Simple NFD Decomposition
//===----------------------------------------------------------------------===//

uint32_t iree_unicode_nfd_base(uint32_t codepoint) {
  // Recursively decompose until we reach ASCII or a character with no further
  // decomposition. This handles characters like Vietnamese ử (U+1EED) which
  // decompose in multiple levels: U+1EED → U+1B0 (ư) → U+0075 (u).
  while (codepoint >= 0x80) {
    const iree_unicode_nfd_mapping_t* mapping =
        iree_unicode_nfd_lookup(codepoint);
    if (!mapping) {
      // No further decomposition available.
      break;
    }
    // Mask off the singleton flag to get the actual base codepoint.
    uint32_t base = mapping->base & IREE_UNICODE_NFD_BASE_MASK;
    if (base == codepoint) {
      // Self-referential mapping (shouldn't happen, but guard against loops).
      break;
    }
    codepoint = base;
  }
  return codepoint;
}

//===----------------------------------------------------------------------===//
// Full NFD Decomposition (including Hangul)
//===----------------------------------------------------------------------===//

// Hangul syllable decomposition constants (Unicode Standard, Chapter 3.12).
// Hangul syllables are algorithmically composed from Jamo components:
//   Syllable = (L * VCount + V) * TCount + T + SBase
// Where L=leading consonant, V=vowel, T=trailing consonant (optional).
#define IREE_UNICODE_HANGUL_S_BASE 0xAC00  // First Hangul syllable: 가
#define IREE_UNICODE_HANGUL_L_BASE 0x1100  // First leading consonant (Choseong)
#define IREE_UNICODE_HANGUL_V_BASE 0x1161  // First vowel (Jungseong): ᅡ
#define IREE_UNICODE_HANGUL_T_BASE 0x11A7  // Trailing consonant base
#define IREE_UNICODE_HANGUL_L_COUNT 19     // Number of leading consonants
#define IREE_UNICODE_HANGUL_V_COUNT 21     // Number of vowels
#define IREE_UNICODE_HANGUL_T_COUNT 28     // Number of trailing consonants
#define IREE_UNICODE_HANGUL_N_COUNT \
  (IREE_UNICODE_HANGUL_V_COUNT * IREE_UNICODE_HANGUL_T_COUNT)  // 588
#define IREE_UNICODE_HANGUL_S_COUNT \
  (IREE_UNICODE_HANGUL_L_COUNT * IREE_UNICODE_HANGUL_N_COUNT)  // 11172

iree_host_size_t iree_unicode_decompose(uint32_t codepoint,
                                        uint32_t* out_codepoints) {
  // ASCII has no decomposition.
  if (codepoint < 0x80) {
    out_codepoints[0] = codepoint;
    return 1;
  }

  // Hangul syllable decomposition (algorithmic).
  // Hangul syllables in U+AC00-U+D7A3 decompose to 2-3 Jamo.
  if (codepoint >= IREE_UNICODE_HANGUL_S_BASE) {
    uint32_t s_index = codepoint - IREE_UNICODE_HANGUL_S_BASE;
    if (s_index < IREE_UNICODE_HANGUL_S_COUNT) {
      // Leading consonant (Choseong).
      out_codepoints[0] =
          IREE_UNICODE_HANGUL_L_BASE + (s_index / IREE_UNICODE_HANGUL_N_COUNT);
      // Vowel (Jungseong).
      out_codepoints[1] = IREE_UNICODE_HANGUL_V_BASE +
                          ((s_index % IREE_UNICODE_HANGUL_N_COUNT) /
                           IREE_UNICODE_HANGUL_T_COUNT);
      // Trailing consonant (Jongseong) - may be absent.
      uint32_t t_index = s_index % IREE_UNICODE_HANGUL_T_COUNT;
      if (t_index > 0) {
        out_codepoints[2] = IREE_UNICODE_HANGUL_T_BASE + t_index;
        return 3;
      }
      return 2;
    }
  }

  // Table-based decomposition for other scripts.
  // Look up the decomposition in the NFD table. The table stores both the base
  // character and the combining mark (if any).
  const iree_unicode_nfd_mapping_t* mapping =
      iree_unicode_nfd_lookup(codepoint);
  if (!mapping) {
    // No decomposition available - return unchanged.
    out_codepoints[0] = codepoint;
    return 1;
  }

  // Get the base codepoint (masking off the singleton flag).
  uint32_t base = mapping->base & IREE_UNICODE_NFD_BASE_MASK;

  // Recursively decompose the base if needed (handles multi-level decomposition
  // like Vietnamese ử: U+1EED → U+01B0 + U+0309, then U+01B0 → u + U+031B).
  iree_host_size_t count = iree_unicode_decompose(base, out_codepoints);

  // Append the combining mark if present.
  if (mapping->combining != 0) {
    out_codepoints[count++] = mapping->combining;
  }

  return count;
}

//===----------------------------------------------------------------------===//
// NFC Normalization
//===----------------------------------------------------------------------===//

// Binary search for codepoint in CCC entries.
static uint8_t iree_unicode_ccc_lookup(uint32_t codepoint) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_ccc_entries_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const iree_unicode_ccc_entry_t* entry = &iree_unicode_ccc_entries[mid];
    if (codepoint < entry->codepoint) {
      high = mid;
    } else if (codepoint > entry->codepoint) {
      low = mid + 1;
    } else {
      return entry->ccc;
    }
  }
  return 0;  // Not found = CCC 0 (starter).
}

uint8_t iree_unicode_ccc(uint32_t codepoint) {
  // ASCII always has CCC 0.
  if (codepoint < 0x80) {
    return 0;
  }
  return iree_unicode_ccc_lookup(codepoint);
}

// Binary search for composition pair.
// The table is sorted by (base, combining) for efficient lookup.
static uint32_t iree_unicode_nfc_pair_lookup(uint32_t base,
                                             uint32_t combining) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_nfc_pairs_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const iree_unicode_nfc_pair_t* pair = &iree_unicode_nfc_pairs[mid];
    if (base < pair->base) {
      high = mid;
    } else if (base > pair->base) {
      low = mid + 1;
    } else if (combining < pair->combining) {
      high = mid;
    } else if (combining > pair->combining) {
      low = mid + 1;
    } else {
      return pair->composed;
    }
  }
  return 0;  // No composition.
}

uint32_t iree_unicode_compose_pair(uint32_t base, uint32_t combining) {
  // Hangul composition (algorithmic, Unicode Standard Chapter 3.12).
  // This is the inverse of the algorithmic decomposition in nfd_decompose.

  // Case 1: L + V → LV syllable.
  // L (leading consonant): U+1100..U+1112
  // V (vowel): U+1161..U+1175
  if (base >= IREE_UNICODE_HANGUL_L_BASE &&
      base < IREE_UNICODE_HANGUL_L_BASE + IREE_UNICODE_HANGUL_L_COUNT) {
    if (combining >= IREE_UNICODE_HANGUL_V_BASE &&
        combining < IREE_UNICODE_HANGUL_V_BASE + IREE_UNICODE_HANGUL_V_COUNT) {
      uint32_t l_index = base - IREE_UNICODE_HANGUL_L_BASE;
      uint32_t v_index = combining - IREE_UNICODE_HANGUL_V_BASE;
      return IREE_UNICODE_HANGUL_S_BASE +
             (l_index * IREE_UNICODE_HANGUL_N_COUNT) +
             (v_index * IREE_UNICODE_HANGUL_T_COUNT);
    }
  }

  // Case 2: LV + T → LVT syllable.
  // LV syllable: a syllable with no trailing consonant (s_index % T_COUNT == 0)
  // T (trailing consonant): U+11A8..U+11C2
  if (base >= IREE_UNICODE_HANGUL_S_BASE) {
    uint32_t s_index = base - IREE_UNICODE_HANGUL_S_BASE;
    if (s_index < IREE_UNICODE_HANGUL_S_COUNT &&
        (s_index % IREE_UNICODE_HANGUL_T_COUNT) == 0) {
      // It's an LV syllable.
      if (combining > IREE_UNICODE_HANGUL_T_BASE &&
          combining <=
              IREE_UNICODE_HANGUL_T_BASE + IREE_UNICODE_HANGUL_T_COUNT - 1) {
        uint32_t t_index = combining - IREE_UNICODE_HANGUL_T_BASE;
        return base + t_index;
      }
    }
  }

  // Fall back to table lookup for non-Hangul compositions.
  return iree_unicode_nfc_pair_lookup(base, combining);
}

// Canonical ordering: sort combining marks by CCC using insertion sort.
// Stable sort is required to preserve order of marks with same CCC.
static void iree_unicode_canonical_order(uint32_t* codepoints,
                                         iree_host_size_t count) {
  for (iree_host_size_t i = 1; i < count; ++i) {
    uint32_t current = codepoints[i];
    uint8_t current_ccc = iree_unicode_ccc(current);
    // Only reorder combining marks (CCC > 0).
    if (current_ccc == 0) continue;
    iree_host_size_t j = i;
    while (j > 0) {
      uint8_t prev_ccc = iree_unicode_ccc(codepoints[j - 1]);
      // Stop if prev is a starter or has lower/equal CCC.
      if (prev_ccc == 0 || prev_ccc <= current_ccc) break;
      codepoints[j] = codepoints[j - 1];
      --j;
    }
    codepoints[j] = current;
  }
}

// Apply canonical composition to a sequence of codepoints.
// Modifies the array in place and returns the new count.
static iree_host_size_t iree_unicode_compose_codepoints(
    uint32_t* codepoints, iree_host_size_t count) {
  if (count < 2) return count;

  iree_host_size_t write_index = 0;
  iree_host_size_t starter_index = 0;
  uint8_t last_ccc = 0;

  // First codepoint.
  codepoints[write_index++] = codepoints[0];
  if (iree_unicode_ccc(codepoints[0]) != 0) {
    // First is not a starter - unusual but handle it.
    starter_index = (iree_host_size_t)-1;
    last_ccc = iree_unicode_ccc(codepoints[0]);
  }

  for (iree_host_size_t i = 1; i < count; ++i) {
    uint32_t current = codepoints[i];
    uint8_t current_ccc = iree_unicode_ccc(current);

    // Try to compose with the starter.
    bool composed = false;
    if (starter_index != (iree_host_size_t)-1) {
      // Can compose if:
      // 1. Current is a starter (CCC 0), or
      // 2. Last CCC < current CCC (not blocked)
      if (current_ccc == 0 || last_ccc < current_ccc) {
        uint32_t composition =
            iree_unicode_compose_pair(codepoints[starter_index], current);
        if (composition != 0) {
          codepoints[starter_index] = composition;
          composed = true;
        }
      }
    }

    if (!composed) {
      if (current_ccc == 0) {
        // New starter.
        starter_index = write_index;
        last_ccc = 0;
      } else {
        last_ccc = current_ccc;
      }
      codepoints[write_index++] = current;
    }
  }

  return write_index;
}

// Maximum combining sequence length (starter + combining marks).
// Unicode Stream-Safe Text Format limits to 30 combining characters.
// Real-world text rarely exceeds 3-4 combining marks per base character.
#define IREE_UNICODE_MAX_COMBINING_SEQUENCE 32

// Flushes a combining sequence: applies canonical ordering and composition,
// then encodes to output. Returns the number of bytes written on success, or
// IREE_HOST_SIZE_MAX if the output buffer would overflow.
static iree_host_size_t iree_unicode_flush_sequence(
    uint32_t* sequence, iree_host_size_t sequence_count, char* out_buffer,
    iree_host_size_t capacity, iree_host_size_t output_position) {
  if (sequence_count == 0) return 0;

  // Apply canonical ordering and composition.
  iree_unicode_canonical_order(sequence, sequence_count);
  iree_host_size_t composed_count =
      iree_unicode_compose_codepoints(sequence, sequence_count);

  // Encode to output buffer.
  iree_host_size_t bytes_written = 0;
  for (iree_host_size_t i = 0; i < composed_count; ++i) {
    iree_host_size_t encoded_length =
        iree_unicode_utf8_encoded_length(sequence[i]);
    if (output_position + bytes_written + encoded_length > capacity) {
      return IREE_HOST_SIZE_MAX;  // Would overflow.
    }
    bytes_written += iree_unicode_utf8_encode(
        sequence[i], out_buffer + output_position + bytes_written);
  }
  return bytes_written;
}

iree_status_t iree_unicode_compose(iree_string_view_t input, char* out_buffer,
                                   iree_host_size_t capacity,
                                   iree_host_size_t* out_length) {
  // Fast path: ASCII-only input needs no composition.
  bool all_ascii = true;
  for (iree_host_size_t i = 0; i < input.size && all_ascii; ++i) {
    if ((uint8_t)input.data[i] > 0x7F) all_ascii = false;
  }
  if (all_ascii) {
    if (input.size > capacity) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    memcpy(out_buffer, input.data, input.size);
    *out_length = input.size;
    return iree_ok_status();
  }

  // Output can only shrink (composition combines characters), so input.size
  // is sufficient capacity.
  if (input.size > capacity) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  // Process combining sequences one at a time using a small fixed buffer.
  // A combining sequence is a starter (CCC=0) followed by combining marks.
  uint32_t sequence[IREE_UNICODE_MAX_COMBINING_SEQUENCE];
  iree_host_size_t sequence_count = 0;
  iree_host_size_t input_position = 0;
  iree_host_size_t output_position = 0;

  while (input_position < input.size) {
    uint32_t codepoint = iree_unicode_utf8_decode(input, &input_position);
    uint8_t ccc = iree_unicode_ccc(codepoint);

    if (ccc == 0 && sequence_count > 0) {
      // New starter: flush the previous sequence.
      iree_host_size_t bytes_written = iree_unicode_flush_sequence(
          sequence, sequence_count, out_buffer, capacity, output_position);
      if (bytes_written == IREE_HOST_SIZE_MAX) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
      output_position += bytes_written;
      sequence_count = 0;
    }

    // Add codepoint to current sequence.
    if (sequence_count >= IREE_UNICODE_MAX_COMBINING_SEQUENCE) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    sequence[sequence_count++] = codepoint;
  }

  // Flush any remaining sequence.
  if (sequence_count > 0) {
    iree_host_size_t bytes_written = iree_unicode_flush_sequence(
        sequence, sequence_count, out_buffer, capacity, output_position);
    if (bytes_written == IREE_HOST_SIZE_MAX) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    output_position += bytes_written;
  }

  *out_length = output_position;
  return iree_ok_status();
}

iree_host_size_t iree_unicode_decompose_singleton(uint32_t codepoint,
                                                  uint32_t* out_codepoints) {
  // Fast path: ASCII has no decomposition.
  if (codepoint < 0x80) {
    out_codepoints[0] = codepoint;
    return 1;
  }

  // Hangul syllable algorithmic decomposition (U+AC00-U+D7A3).
  // Decomposes to 2-3 Jamo which will be recomposed by NFC composition step.
  if (codepoint >= IREE_UNICODE_HANGUL_S_BASE &&
      codepoint < IREE_UNICODE_HANGUL_S_BASE + IREE_UNICODE_HANGUL_S_COUNT) {
    uint32_t s_index = codepoint - IREE_UNICODE_HANGUL_S_BASE;
    out_codepoints[0] =
        IREE_UNICODE_HANGUL_L_BASE + (s_index / IREE_UNICODE_HANGUL_N_COUNT);
    out_codepoints[1] =
        IREE_UNICODE_HANGUL_V_BASE +
        ((s_index % IREE_UNICODE_HANGUL_N_COUNT) / IREE_UNICODE_HANGUL_T_COUNT);
    uint32_t t_index = s_index % IREE_UNICODE_HANGUL_T_COUNT;
    if (t_index > 0) {
      out_codepoints[2] = IREE_UNICODE_HANGUL_T_BASE + t_index;
      return 3;
    }
    return 2;
  }

  // Check table for singleton decomposition only (e.g., CJK Compatibility).
  // Non-singleton decompositions (like é) are NOT applied since those
  // characters are already in NFC form.
  const iree_unicode_nfd_mapping_t* mapping =
      iree_unicode_nfd_lookup(codepoint);
  if (mapping && (mapping->base & IREE_UNICODE_NFD_SINGLETON_FLAG)) {
    out_codepoints[0] = mapping->base & IREE_UNICODE_NFD_BASE_MASK;
    return 1;
  }

  // No decomposition needed - return unchanged.
  out_codepoints[0] = codepoint;
  return 1;
}

// Binary search for an NFC decomposition entry by codepoint.
static const iree_unicode_nfc_decomp_t* iree_unicode_nfc_decomp_lookup(
    uint32_t codepoint) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_nfc_decompositions_count;
  while (low < high) {
    iree_host_size_t mid = (low + high) / 2;
    const iree_unicode_nfc_decomp_t* entry =
        &iree_unicode_nfc_decompositions[mid];
    if (entry->codepoint < codepoint) {
      low = mid + 1;
    } else if (entry->codepoint > codepoint) {
      high = mid;
    } else {
      return entry;
    }
  }
  return NULL;
}

iree_host_size_t iree_unicode_decompose_nfc_canonical(
    uint32_t codepoint, uint32_t* out_codepoints) {
  // Fast path: ASCII has no decomposition.
  if (codepoint < 0x80) {
    out_codepoints[0] = codepoint;
    return 1;
  }

  // Hangul syllable algorithmic decomposition (U+AC00-U+D7A3).
  if (codepoint >= IREE_UNICODE_HANGUL_S_BASE &&
      codepoint < IREE_UNICODE_HANGUL_S_BASE + IREE_UNICODE_HANGUL_S_COUNT) {
    uint32_t s_index = codepoint - IREE_UNICODE_HANGUL_S_BASE;
    out_codepoints[0] =
        IREE_UNICODE_HANGUL_L_BASE + (s_index / IREE_UNICODE_HANGUL_N_COUNT);
    out_codepoints[1] =
        IREE_UNICODE_HANGUL_V_BASE +
        ((s_index % IREE_UNICODE_HANGUL_N_COUNT) / IREE_UNICODE_HANGUL_T_COUNT);
    uint32_t t_index = s_index % IREE_UNICODE_HANGUL_T_COUNT;
    if (t_index > 0) {
      out_codepoints[2] = IREE_UNICODE_HANGUL_T_BASE + t_index;
      return 3;
    }
    return 2;
  }

  // Check the NFC_QC=No decomposition table (covers singleton and
  // multi-codepoint canonical decompositions not handled elsewhere).
  const iree_unicode_nfc_decomp_t* decomp =
      iree_unicode_nfc_decomp_lookup(codepoint);
  if (decomp) {
    iree_host_size_t count = 0;
    for (iree_host_size_t i = 0; i < 3 && decomp->target[i] != 0; ++i) {
      out_codepoints[count++] = decomp->target[i];
    }
    return count;
  }

  // Check the existing NFD singleton table (CJK Compatibility Ideographs, etc.)
  const iree_unicode_nfd_mapping_t* mapping =
      iree_unicode_nfd_lookup(codepoint);
  if (mapping && (mapping->base & IREE_UNICODE_NFD_SINGLETON_FLAG)) {
    out_codepoints[0] = mapping->base & IREE_UNICODE_NFD_BASE_MASK;
    return 1;
  }

  // No decomposition needed - return unchanged.
  out_codepoints[0] = codepoint;
  return 1;
}

iree_status_t iree_unicode_nfc(iree_string_view_t input,
                               iree_host_size_t out_capacity, char* out_buffer,
                               iree_host_size_t* out_length) {
  // Fast path: ASCII-only input needs no normalization.
  bool all_ascii = true;
  for (iree_host_size_t i = 0; i < input.size && all_ascii; ++i) {
    if ((uint8_t)input.data[i] > 0x7F) all_ascii = false;
  }
  if (all_ascii) {
    if (input.size > out_capacity) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "NFC output buffer too small for ASCII input (need %" PRIhsz
          ", have %" PRIhsz ")",
          input.size, out_capacity);
    }
    memcpy(out_buffer, input.data, input.size);
    *out_length = input.size;
    return iree_ok_status();
  }

  // Unlike compose(), NFC can expand intermediate output (Hangul syllables
  // decompose to 2-3 Jamo), but after composition the output typically shrinks.
  // We need enough capacity for the intermediate decomposed form.
  // Worst case: every input byte becomes 4 output bytes (if all Hangul).
  // In practice, most text is already NFC so little expansion occurs.

  // Process combining sequences one at a time using a small fixed buffer.
  // A combining sequence is a starter (CCC=0) followed by combining marks.
  uint32_t sequence[IREE_UNICODE_MAX_COMBINING_SEQUENCE];
  iree_host_size_t sequence_count = 0;
  iree_host_size_t input_position = 0;
  iree_host_size_t output_position = 0;

  while (input_position < input.size) {
    uint32_t codepoint = iree_unicode_utf8_decode(input, &input_position);

    // Canonical decomposition for NFC: decomposes Hangul, CJK compatibility,
    // and all NFC_QC=No characters. Precomposed characters like é are left
    // unchanged (they are already in NFC form).
    uint32_t decomposed[4];
    iree_host_size_t decomposed_count =
        iree_unicode_decompose_nfc_canonical(codepoint, decomposed);

    // Process each decomposed codepoint.
    for (iree_host_size_t d = 0; d < decomposed_count; ++d) {
      uint32_t cp = decomposed[d];
      uint8_t ccc = iree_unicode_ccc(cp);

      if (ccc == 0 && sequence_count > 0) {
        // New starter encountered. Before flushing, try to compose with the
        // last codepoint in the sequence. This handles Hangul Jamo composition
        // where L+V and LV+T are both compositions of two starters (CCC 0).
        uint32_t last_cp = sequence[sequence_count - 1];
        uint32_t composition = iree_unicode_compose_pair(last_cp, cp);
        if (composition != 0) {
          // Composition succeeded - replace last codepoint, skip adding cp.
          sequence[sequence_count - 1] = composition;
          continue;
        }

        // Composition failed - flush the previous sequence and start fresh.
        iree_host_size_t bytes_written =
            iree_unicode_flush_sequence(sequence, sequence_count, out_buffer,
                                        out_capacity, output_position);
        if (bytes_written == IREE_HOST_SIZE_MAX) {
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "NFC output buffer overflow while flushing sequence at position "
              "%" PRIhsz,
              output_position);
        }
        output_position += bytes_written;
        sequence_count = 0;
      }

      // Add decomposed codepoint to current sequence.
      if (sequence_count >= IREE_UNICODE_MAX_COMBINING_SEQUENCE) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "NFC combining sequence exceeds maximum length "
                                "(%d codepoints) at input position %" PRIhsz,
                                IREE_UNICODE_MAX_COMBINING_SEQUENCE,
                                input_position);
      }
      sequence[sequence_count++] = cp;
    }
  }

  // Flush any remaining sequence.
  if (sequence_count > 0) {
    iree_host_size_t bytes_written = iree_unicode_flush_sequence(
        sequence, sequence_count, out_buffer, out_capacity, output_position);
    if (bytes_written == IREE_HOST_SIZE_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "NFC output buffer overflow while flushing final sequence at "
          "position %" PRIhsz,
          output_position);
    }
    output_position += bytes_written;
  }

  *out_length = output_position;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// NFKD Decomposition
//===----------------------------------------------------------------------===//

// Binary search for codepoint in NFKD mappings.
static const iree_unicode_nfkd_mapping_t* iree_unicode_nfkd_lookup(
    uint32_t codepoint) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_nfkd_mappings_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const iree_unicode_nfkd_mapping_t* mapping =
        &iree_unicode_nfkd_mappings[mid];
    if (codepoint < mapping->codepoint) {
      high = mid;
    } else if (codepoint > mapping->codepoint) {
      low = mid + 1;
    } else {
      return mapping;
    }
  }
  return NULL;
}

iree_host_size_t iree_unicode_decompose_nfkd(uint32_t codepoint,
                                             uint32_t* out_codepoints) {
  // ASCII has no decomposition.
  if (codepoint < 0x80) {
    out_codepoints[0] = codepoint;
    return 1;
  }

  // Look up NFKD compatibility decomposition.
  const iree_unicode_nfkd_mapping_t* mapping =
      iree_unicode_nfkd_lookup(codepoint);
  if (mapping) {
    // Found a compatibility decomposition.
    // The table already contains the fully-expanded NFKD form (compatibility +
    // canonical decompositions applied recursively by the generator).
    iree_host_size_t length = mapping->length;
    if (length <= 4) {
      // Inline storage.
      for (iree_host_size_t i = 0; i < length; ++i) {
        out_codepoints[i] = mapping->inline_targets[i];
      }
    } else {
      // Overflow storage.
      const uint32_t* targets = iree_unicode_nfkd_overflow + mapping->offset;
      for (iree_host_size_t i = 0; i < length; ++i) {
        out_codepoints[i] = targets[i];
      }
    }
    return length;
  }

  // No compatibility decomposition - apply canonical (NFD) decomposition.
  // This handles:
  // - Hangul syllables (algorithmic decomposition)
  // - Precomposed characters like é → e + combining acute
  return iree_unicode_decompose(codepoint, out_codepoints);
}
