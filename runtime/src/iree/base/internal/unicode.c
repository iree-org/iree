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

extern const iree_unicode_case_mapping_t iree_unicode_case_mappings[];
extern const iree_host_size_t iree_unicode_case_mappings_count;

extern const iree_unicode_nfd_mapping_t iree_unicode_nfd_mappings[];
extern const iree_host_size_t iree_unicode_nfd_mappings_count;

extern const iree_unicode_ccc_entry_t iree_unicode_ccc_entries[];
extern const iree_host_size_t iree_unicode_ccc_entries_count;

extern const iree_unicode_nfc_pair_t iree_unicode_nfc_pairs[];
extern const iree_host_size_t iree_unicode_nfc_pairs_count;

//===----------------------------------------------------------------------===//
// UTF-8 Codec
//===----------------------------------------------------------------------===//

uint32_t iree_unicode_utf8_decode(iree_string_view_t text,
                                  iree_host_size_t* position) {
  if (*position >= text.size) {
    return IREE_UNICODE_REPLACEMENT_CHAR;
  }

  const uint8_t* data = (const uint8_t*)text.data;
  uint8_t first_byte = data[*position];

  // Single byte (ASCII): 0xxxxxxx
  if ((first_byte & 0x80) == 0) {
    (*position)++;
    return first_byte;
  }

  // Determine sequence length from leading byte.
  iree_host_size_t sequence_length;
  uint32_t codepoint;
  if ((first_byte & 0xE0) == 0xC0) {
    // Two bytes: 110xxxxx 10xxxxxx
    sequence_length = 2;
    codepoint = first_byte & 0x1F;
  } else if ((first_byte & 0xF0) == 0xE0) {
    // Three bytes: 1110xxxx 10xxxxxx 10xxxxxx
    sequence_length = 3;
    codepoint = first_byte & 0x0F;
  } else if ((first_byte & 0xF8) == 0xF0) {
    // Four bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    sequence_length = 4;
    codepoint = first_byte & 0x07;
  } else {
    // Invalid leading byte.
    (*position)++;
    return IREE_UNICODE_REPLACEMENT_CHAR;
  }

  // Check we have enough bytes remaining.
  if (*position + sequence_length > text.size) {
    (*position)++;
    return IREE_UNICODE_REPLACEMENT_CHAR;
  }

  // Decode continuation bytes.
  for (iree_host_size_t i = 1; i < sequence_length; ++i) {
    uint8_t continuation_byte = data[*position + i];
    if ((continuation_byte & 0xC0) != 0x80) {
      // Invalid continuation byte.
      (*position)++;
      return IREE_UNICODE_REPLACEMENT_CHAR;
    }
    codepoint = (codepoint << 6) | (continuation_byte & 0x3F);
  }

  // Validate codepoint range and overlong sequences.
  bool valid = true;
  if (codepoint > IREE_UNICODE_MAX_CODEPOINT) {
    valid = false;
  } else if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
    // Surrogate pairs are invalid in UTF-8.
    valid = false;
  } else if (sequence_length == 2 && codepoint < 0x80) {
    // Overlong 2-byte sequence.
    valid = false;
  } else if (sequence_length == 3 && codepoint < 0x800) {
    // Overlong 3-byte sequence.
    valid = false;
  } else if (sequence_length == 4 && codepoint < 0x10000) {
    // Overlong 4-byte sequence.
    valid = false;
  }

  if (!valid) {
    (*position)++;
    return IREE_UNICODE_REPLACEMENT_CHAR;
  }

  *position += sequence_length;
  return codepoint;
}

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

    // Found a lead byte. Determine expected sequence length.
    iree_host_size_t expected_length;
    if ((byte & 0x80) == 0x00) {
      // ASCII (0xxxxxxx) - 1 byte, always complete.
      expected_length = 1;
    } else if ((byte & 0xE0) == 0xC0) {
      // 2-byte sequence (110xxxxx).
      expected_length = 2;
    } else if ((byte & 0xF0) == 0xE0) {
      // 3-byte sequence (1110xxxx).
      expected_length = 3;
    } else if ((byte & 0xF8) == 0xF0) {
      // 4-byte sequence (11110xxx).
      expected_length = 4;
    } else {
      // Invalid lead byte (0xFF, 0xFE, or overlong) - treat as complete.
      return 0;
    }

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
static iree_unicode_category_t iree_unicode_category_lookup(
    uint32_t codepoint) {
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
static bool iree_unicode_whitespace_lookup(uint32_t codepoint) {
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

// Binary search for codepoint in case mappings.
static const iree_unicode_case_mapping_t* iree_unicode_case_lookup(
    uint32_t codepoint) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_unicode_case_mappings_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const iree_unicode_case_mapping_t* mapping =
        &iree_unicode_case_mappings[mid];
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

iree_unicode_category_t iree_unicode_category(uint32_t codepoint) {
  // Fast path for ASCII.
  if (codepoint < 0x80) {
    if (codepoint >= 'A' && codepoint <= 'Z') {
      return IREE_UNICODE_CATEGORY_LETTER;
    }
    if (codepoint >= 'a' && codepoint <= 'z') {
      return IREE_UNICODE_CATEGORY_LETTER;
    }
    if (codepoint >= '0' && codepoint <= '9') {
      return IREE_UNICODE_CATEGORY_NUMBER;
    }
    if (codepoint < 0x20 || codepoint == 0x7F) {
      return IREE_UNICODE_CATEGORY_OTHER;  // Control
    }
    if (codepoint == ' ') {
      return IREE_UNICODE_CATEGORY_SEPARATOR;
    }
    // ASCII punctuation and symbols.
    if ((codepoint >= '!' && codepoint <= '/') ||
        (codepoint >= ':' && codepoint <= '@') ||
        (codepoint >= '[' && codepoint <= '`') ||
        (codepoint >= '{' && codepoint <= '~')) {
      // Mix of punctuation and symbols - check specific ranges.
      if (codepoint == '$' || codepoint == '+' || codepoint == '<' ||
          codepoint == '=' || codepoint == '>' || codepoint == '^' ||
          codepoint == '`' || codepoint == '|' || codepoint == '~') {
        return IREE_UNICODE_CATEGORY_SYMBOL;
      }
      return IREE_UNICODE_CATEGORY_PUNCTUATION;
    }
  }
  return iree_unicode_category_lookup(codepoint);
}

bool iree_unicode_is_whitespace(uint32_t codepoint) {
  // Fast path for common whitespace.
  if (codepoint == ' ' || codepoint == '\t' || codepoint == '\n' ||
      codepoint == '\r' || codepoint == '\f' || codepoint == '\v') {
    return true;
  }
  // Check the full whitespace table for non-ASCII.
  if (codepoint >= 0x80) {
    return iree_unicode_whitespace_lookup(codepoint);
  }
  return false;
}

bool iree_unicode_is_control(uint32_t codepoint) {
  // Cc category: C0 controls (0x00-0x1F), DEL (0x7F), C1 controls (0x80-0x9F).
  return codepoint <= 0x1F || (codepoint >= 0x7F && codepoint <= 0x9F);
}

bool iree_unicode_is_cjk(uint32_t codepoint) {
  // CJK Unified Ideographs and related blocks.
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

uint32_t iree_unicode_to_lower(uint32_t codepoint) {
  // Fast path for ASCII.
  if (codepoint >= 'A' && codepoint <= 'Z') {
    return codepoint + ('a' - 'A');
  }
  if (codepoint < 0x80) {
    return codepoint;
  }
  const iree_unicode_case_mapping_t* mapping =
      iree_unicode_case_lookup(codepoint);
  if (mapping && mapping->lowercase != 0) {
    return mapping->lowercase;
  }
  return codepoint;
}

uint32_t iree_unicode_to_upper(uint32_t codepoint) {
  // Fast path for ASCII.
  if (codepoint >= 'a' && codepoint <= 'z') {
    return codepoint - ('a' - 'A');
  }
  if (codepoint < 0x80) {
    return codepoint;
  }
  const iree_unicode_case_mapping_t* mapping =
      iree_unicode_case_lookup(codepoint);
  if (mapping && mapping->uppercase != 0) {
    return mapping->uppercase;
  }
  return codepoint;
}

//===----------------------------------------------------------------------===//
// Simple NFD Decomposition
//===----------------------------------------------------------------------===//

uint32_t iree_unicode_nfd_base(uint32_t codepoint) {
  // ASCII has no decomposition.
  if (codepoint < 0x80) {
    return codepoint;
  }
  const iree_unicode_nfd_mapping_t* mapping =
      iree_unicode_nfd_lookup(codepoint);
  if (mapping) {
    return mapping->base;
  }
  return codepoint;
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
// then encodes to output. Returns the number of bytes written, or 0 on error.
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
      return 0;  // Would overflow.
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
      if (bytes_written == 0 && sequence_count > 0) {
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
    if (bytes_written == 0 && sequence_count > 0) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    output_position += bytes_written;
  }

  *out_length = output_position;
  return iree_ok_status();
}
