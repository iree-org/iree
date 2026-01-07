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
