// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/regex/internal/lexer.h"

#include <ctype.h>
#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/regex/exec.h"

//===----------------------------------------------------------------------===//
// Helper Macros
//===----------------------------------------------------------------------===//

// Set bitmap bit for byte value.
#define BITMAP_SET(bitmap, byte) ((bitmap)[(byte) >> 3] |= (1u << ((byte) & 7)))

// Test bitmap bit for byte value.
#define BITMAP_TEST(bitmap, byte) \
  (((bitmap)[(byte) >> 3] & (1u << ((byte) & 7))) != 0)

//===----------------------------------------------------------------------===//
// Character Class Range Utilities
//===----------------------------------------------------------------------===//

// Inserts a codepoint range into the ranges array, maintaining sorted order
// by start codepoint. Returns true on success, false if the array is full.
// Sorted ranges enable early-exit optimization: if codepoint < ranges[i].start,
// it cannot match any subsequent range.
static bool iree_tokenizer_regex_insert_sorted_range(
    iree_tokenizer_regex_codepoint_range_t* ranges, uint8_t* range_count,
    uint32_t start, uint32_t end) {
  if (*range_count >= IREE_TOKENIZER_REGEX_MAX_CHAR_CLASS_RANGES) {
    return false;
  }

  // Find insertion point to maintain sorted order by start.
  uint8_t insert_pos = 0;
  while (insert_pos < *range_count && ranges[insert_pos].start < start) {
    ++insert_pos;
  }

  // Shift existing ranges to make room.
  for (uint8_t i = *range_count; i > insert_pos; --i) {
    ranges[i] = ranges[i - 1];
  }

  // Insert new range.
  ranges[insert_pos].start = start;
  ranges[insert_pos].end = end;
  ++(*range_count);
  return true;
}

// Adds a single codepoint to the character class.
// ASCII codepoints (< 0x80) are added to the bitmap.
// Non-ASCII codepoints are added as single-element ranges for exact matching.
// Returns true on success, false if adding a non-ASCII codepoint exceeds the
// 4-range limit.
static bool iree_tokenizer_regex_char_class_add_codepoint(
    uint8_t* bitmap, iree_tokenizer_regex_codepoint_range_t* ranges,
    uint8_t* range_count, uint32_t codepoint) {
  if (codepoint < 0x80) {
    BITMAP_SET(bitmap, codepoint);
    return true;
  }
  // Non-ASCII: use exact codepoint range for precise matching.
  return iree_tokenizer_regex_insert_sorted_range(ranges, range_count,
                                                  codepoint, codepoint);
}

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

static void iree_tokenizer_regex_lexer_scan(
    iree_tokenizer_regex_lexer_t* lexer);

//===----------------------------------------------------------------------===//
// Lexer Initialization
//===----------------------------------------------------------------------===//

void iree_tokenizer_regex_lexer_initialize(iree_tokenizer_regex_lexer_t* lexer,
                                           iree_string_view_t pattern) {
  memset(lexer, 0, sizeof(*lexer));
  lexer->input = pattern;
  lexer->position = 0;
  lexer->has_peeked = false;
}

//===----------------------------------------------------------------------===//
// Lexer API
//===----------------------------------------------------------------------===//

const iree_tokenizer_regex_token_t* iree_tokenizer_regex_lexer_peek(
    iree_tokenizer_regex_lexer_t* lexer) {
  if (!lexer->has_peeked) {
    iree_tokenizer_regex_lexer_scan(lexer);
    lexer->has_peeked = true;
  }
  return &lexer->current;
}

void iree_tokenizer_regex_lexer_advance(iree_tokenizer_regex_lexer_t* lexer) {
  if (!lexer->has_peeked) {
    iree_tokenizer_regex_lexer_scan(lexer);
  }
  lexer->has_peeked = false;
}

iree_host_size_t iree_tokenizer_regex_lexer_position(
    const iree_tokenizer_regex_lexer_t* lexer) {
  return lexer->position;
}

//===----------------------------------------------------------------------===//
// Internal Helpers
//===----------------------------------------------------------------------===//

// Peeks at the current character without advancing.
static inline char iree_tokenizer_regex_lexer_peek_char(
    const iree_tokenizer_regex_lexer_t* lexer) {
  if (lexer->position >= lexer->input.size) return '\0';
  return lexer->input.data[lexer->position];
}

// Peeks at the next character (position + 1).
static inline char iree_tokenizer_regex_lexer_peek_char_at(
    const iree_tokenizer_regex_lexer_t* lexer, iree_host_size_t offset) {
  iree_host_size_t position = lexer->position + offset;
  if (position >= lexer->input.size) return '\0';
  return lexer->input.data[position];
}

// Advances position by one character.
static inline void iree_tokenizer_regex_lexer_advance_char(
    iree_tokenizer_regex_lexer_t* lexer) {
  if (lexer->position < lexer->input.size) {
    ++lexer->position;
  }
}

// Sets the current token to an error.
static void iree_tokenizer_regex_lexer_set_error(
    iree_tokenizer_regex_lexer_t* lexer, iree_host_size_t position,
    const char* message) {
  lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_ERROR;
  lexer->current.position = position;
  lexer->current.length = 1;
  lexer->current.value.error_message = message;
}

// Sets the current token to a simple token type.
static void iree_tokenizer_regex_lexer_set_simple(
    iree_tokenizer_regex_lexer_t* lexer, iree_tokenizer_regex_token_type_t type,
    iree_host_size_t position, iree_host_size_t length) {
  lexer->current.type = type;
  lexer->current.position = position;
  lexer->current.length = length;
}

// Sets the current token to a literal.
static void iree_tokenizer_regex_lexer_set_literal(
    iree_tokenizer_regex_lexer_t* lexer, uint8_t byte,
    iree_host_size_t position, iree_host_size_t length) {
  lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_LITERAL;
  lexer->current.position = position;
  lexer->current.length = length;
  lexer->current.value.literal = byte;
}

//===----------------------------------------------------------------------===//
// Hex Digit Parsing
//===----------------------------------------------------------------------===//

// Converts a hex character to its numeric value (0-15), or -1 if invalid.
static inline int iree_tokenizer_regex_hex_digit_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Parses |count| hex digits from lexer, advancing position.
// Returns the parsed value, or -1 if not enough valid hex digits.
static int iree_tokenizer_regex_lexer_parse_hex_digits(
    iree_tokenizer_regex_lexer_t* lexer, int count) {
  int value = 0;
  for (int i = 0; i < count; i++) {
    char c = iree_tokenizer_regex_lexer_peek_char(lexer);
    int digit = iree_tokenizer_regex_hex_digit_value(c);
    if (digit < 0) return -1;
    value = (value << 4) | digit;
    iree_tokenizer_regex_lexer_advance_char(lexer);
  }
  return value;
}

//===----------------------------------------------------------------------===//
// Escape Sequence Handling
//===----------------------------------------------------------------------===//

// Parses an escape sequence starting after the backslash.
// Returns true on success, sets token and advances position.
static bool iree_tokenizer_regex_lexer_parse_escape(
    iree_tokenizer_regex_lexer_t* lexer, iree_host_size_t start_pos) {
  char c = iree_tokenizer_regex_lexer_peek_char(lexer);
  if (c == '\0') {
    iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                         "trailing backslash");
    return false;
  }

  iree_tokenizer_regex_lexer_advance_char(lexer);
  iree_host_size_t length = lexer->position - start_pos;

  switch (c) {
    // Simple escapes.
    case 'n':
      iree_tokenizer_regex_lexer_set_literal(lexer, '\n', start_pos, length);
      return true;
    case 'r':
      iree_tokenizer_regex_lexer_set_literal(lexer, '\r', start_pos, length);
      return true;
    case 't':
      iree_tokenizer_regex_lexer_set_literal(lexer, '\t', start_pos, length);
      return true;
    case 'f':
      iree_tokenizer_regex_lexer_set_literal(lexer, '\f', start_pos, length);
      return true;
    case 'v':
      iree_tokenizer_regex_lexer_set_literal(lexer, '\v', start_pos, length);
      return true;
    case 'a':
      iree_tokenizer_regex_lexer_set_literal(lexer, '\a', start_pos, length);
      return true;
    case 'e':
      iree_tokenizer_regex_lexer_set_literal(lexer, 0x1B, start_pos,
                                             length);  // ESC
      return true;
    case '0':
      iree_tokenizer_regex_lexer_set_literal(lexer, '\0', start_pos, length);
      return true;

    // Escaped meta-characters (become literals).
    case '\\':
    case '.':
    case '^':
    case '$':
    case '|':
    case '*':
    case '+':
    case '?':
    case '(':
    case ')':
    case '[':
    case ']':
    case '{':
    case '}':
    case '-':
    case '/':
      iree_tokenizer_regex_lexer_set_literal(lexer, (uint8_t)c, start_pos,
                                             length);
      return true;

    // Shorthand classes.
    case 'd':
      lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND;
      lexer->current.position = start_pos;
      lexer->current.length = length;
      lexer->current.value.shorthand = IREE_TOKENIZER_REGEX_SHORTHAND_d;
      return true;
    case 'D':
      lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND;
      lexer->current.position = start_pos;
      lexer->current.length = length;
      lexer->current.value.shorthand = IREE_TOKENIZER_REGEX_SHORTHAND_D;
      return true;
    case 'w':
      lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND;
      lexer->current.position = start_pos;
      lexer->current.length = length;
      lexer->current.value.shorthand = IREE_TOKENIZER_REGEX_SHORTHAND_w;
      return true;
    case 'W':
      lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND;
      lexer->current.position = start_pos;
      lexer->current.length = length;
      lexer->current.value.shorthand = IREE_TOKENIZER_REGEX_SHORTHAND_W;
      return true;
    case 's':
      lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND;
      lexer->current.position = start_pos;
      lexer->current.length = length;
      lexer->current.value.shorthand = IREE_TOKENIZER_REGEX_SHORTHAND_s;
      return true;
    case 'S':
      lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND;
      lexer->current.position = start_pos;
      lexer->current.length = length;
      lexer->current.value.shorthand = IREE_TOKENIZER_REGEX_SHORTHAND_S;
      return true;

    // Unicode property \p{...}.
    case 'p':
    case 'P': {
      bool negated = (c == 'P');
      if (iree_tokenizer_regex_lexer_peek_char(lexer) != '{') {
        iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                             "expected '{' after \\p");
        return false;
      }
      iree_tokenizer_regex_lexer_advance_char(lexer);  // Skip '{'.

      // Read property name.
      iree_host_size_t prop_start = lexer->position;
      while (iree_tokenizer_regex_lexer_peek_char(lexer) != '}' &&
             iree_tokenizer_regex_lexer_peek_char(lexer) != '\0') {
        iree_tokenizer_regex_lexer_advance_char(lexer);
      }
      if (iree_tokenizer_regex_lexer_peek_char(lexer) != '}') {
        iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                             "unterminated \\p{...}");
        return false;
      }
      iree_host_size_t prop_end = lexer->position;
      iree_tokenizer_regex_lexer_advance_char(lexer);  // Skip '}'.

      iree_host_size_t prop_length = prop_end - prop_start;
      const char* prop = lexer->input.data + prop_start;

      // Map property to pseudo-byte.
      uint8_t pseudo = 0;
      if (prop_length == 1) {
        switch (prop[0]) {
          case 'L':
            pseudo = IREE_TOKENIZER_REGEX_PSEUDO_LETTER;
            break;
          case 'N':
            pseudo = IREE_TOKENIZER_REGEX_PSEUDO_NUMBER;
            break;
          case 'P':
            pseudo = IREE_TOKENIZER_REGEX_PSEUDO_PUNCT;
            break;
          case 'M':
            pseudo = IREE_TOKENIZER_REGEX_PSEUDO_MARK;
            break;
          case 'S':
            pseudo = IREE_TOKENIZER_REGEX_PSEUDO_SYMBOL;
            break;
          case 'Z':
            pseudo = IREE_TOKENIZER_REGEX_PSEUDO_SEPARATOR;
            break;
          case 'C':
            pseudo = IREE_TOKENIZER_REGEX_PSEUDO_OTHER;
            break;
          default:
            iree_tokenizer_regex_lexer_set_error(lexer, prop_start,
                                                 "unknown Unicode property");
            return false;
        }
      } else {
        // Unicode subcategories (Lu, Ll, Nd, etc.) are not supported.
        // Only general categories (L, N, P, M, S, Z, C) are available.
        // Use \p{L} instead of \p{Lu}, etc.
        iree_tokenizer_regex_lexer_set_error(lexer, prop_start,
                                             "unknown Unicode property");
        return false;
      }

      // For \P{...}, we need to negate. This is complex because we'd need
      // to set all OTHER pseudo-bytes. For now, we only support \p{}, not \P{}.
      if (negated) {
        iree_tokenizer_regex_lexer_set_error(
            lexer, start_pos, "\\P{} not supported, use [^\\p{}]");
        return false;
      }

      lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_UNICODE_PROP;
      lexer->current.position = start_pos;
      lexer->current.length = lexer->position - start_pos;
      lexer->current.value.unicode_pseudo_byte = pseudo;
      return true;
    }

    // \uXXXX Unicode escape.
    case 'u': {
      int codepoint = iree_tokenizer_regex_lexer_parse_hex_digits(lexer, 4);
      if (codepoint < 0) {
        iree_tokenizer_regex_lexer_set_error(
            lexer, start_pos, "invalid \\u escape: expected 4 hex digits");
        return false;
      }
      if (codepoint > IREE_UNICODE_MAX_CODEPOINT) {
        iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                             "invalid Unicode codepoint");
        return false;
      }
      // Surrogate codepoints (0xD800-0xDFFF) are not valid Unicode scalars.
      // They are reserved for UTF-16 encoding pairs and cannot appear alone.
      if ((codepoint & 0xFFFFF800) == 0xD800) {
        iree_tokenizer_regex_lexer_set_error(
            lexer, start_pos,
            "surrogate codepoints (U+D800-U+DFFF) not allowed");
        return false;
      }

      // ASCII codepoints become literals.
      if (codepoint < 0x80) {
        iree_tokenizer_regex_lexer_set_literal(
            lexer, (uint8_t)codepoint, start_pos, lexer->position - start_pos);
        return true;
      }

      // Non-ASCII: emit as CHAR_CLASS with single-element range for exact
      // matching. This ensures \u4E2D matches only 中, not all CJK characters.
      memset(&lexer->current.value.char_class, 0,
             sizeof(lexer->current.value.char_class));
      lexer->current.value.char_class.ranges[0].start = (uint32_t)codepoint;
      lexer->current.value.char_class.ranges[0].end = (uint32_t)codepoint;
      lexer->current.value.char_class.range_count = 1;
      lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_CHAR_CLASS;
      lexer->current.position = start_pos;
      lexer->current.length = lexer->position - start_pos;
      return true;
    }

    default:
      // Unknown escape - reject with error for strict validation.
      iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                           "unknown escape sequence");
      return false;
  }
}

//===----------------------------------------------------------------------===//
// Character Class Handling
//===----------------------------------------------------------------------===//

// Adds a shorthand class to the bitmap.
static void iree_tokenizer_regex_bitmap_add_shorthand(
    uint8_t* bitmap, uint16_t* pseudo_mask,
    iree_tokenizer_regex_shorthand_t shorthand) {
  switch (shorthand) {
    case IREE_TOKENIZER_REGEX_SHORTHAND_d:
      // [0-9]
      for (int c = '0'; c <= '9'; ++c) BITMAP_SET(bitmap, c);
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_D:
      // [^0-9] - set all except digits
      memset(bitmap, 0xFF, 32);
      for (int c = '0'; c <= '9'; ++c) bitmap[c >> 3] &= ~(1u << (c & 7));
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_w:
      // [a-zA-Z0-9_]
      for (int c = 'a'; c <= 'z'; ++c) BITMAP_SET(bitmap, c);
      for (int c = 'A'; c <= 'Z'; ++c) BITMAP_SET(bitmap, c);
      for (int c = '0'; c <= '9'; ++c) BITMAP_SET(bitmap, c);
      BITMAP_SET(bitmap, '_');
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_W:
      // [^a-zA-Z0-9_]
      memset(bitmap, 0xFF, 32);
      for (int c = 'a'; c <= 'z'; ++c) bitmap[c >> 3] &= ~(1u << (c & 7));
      for (int c = 'A'; c <= 'Z'; ++c) bitmap[c >> 3] &= ~(1u << (c & 7));
      for (int c = '0'; c <= '9'; ++c) bitmap[c >> 3] &= ~(1u << (c & 7));
      bitmap['_' >> 3] &= ~(1u << ('_' & 7));
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_s:
      // [ \t\r\n\f\v] + Unicode whitespace via pseudo-byte
      BITMAP_SET(bitmap, ' ');
      BITMAP_SET(bitmap, '\t');
      BITMAP_SET(bitmap, '\r');
      BITMAP_SET(bitmap, '\n');
      BITMAP_SET(bitmap, '\f');
      BITMAP_SET(bitmap, '\v');
      // Add pseudo-byte for non-ASCII whitespace.
      *pseudo_mask |= (1u << (IREE_TOKENIZER_REGEX_PSEUDO_WHITESPACE - 0x80));
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_S:
      // [^ \t\r\n\f\v] - everything except whitespace
      memset(bitmap, 0xFF, 32);
      bitmap[' ' >> 3] &= ~(1u << (' ' & 7));
      bitmap['\t' >> 3] &= ~(1u << ('\t' & 7));
      bitmap['\r' >> 3] &= ~(1u << ('\r' & 7));
      bitmap['\n' >> 3] &= ~(1u << ('\n' & 7));
      bitmap['\f' >> 3] &= ~(1u << ('\f' & 7));
      bitmap['\v' >> 3] &= ~(1u << ('\v' & 7));
      // Clear the whitespace pseudo-byte, set all others.
      *pseudo_mask =
          0xFF & ~(1u << (IREE_TOKENIZER_REGEX_PSEUDO_WHITESPACE - 0x80));
      break;
  }
}

// Parses a character class [...].
// Assumes position is at the opening '['.
static bool iree_tokenizer_regex_lexer_parse_char_class(
    iree_tokenizer_regex_lexer_t* lexer) {
  iree_host_size_t start_pos = lexer->position;
  iree_tokenizer_regex_lexer_advance_char(lexer);  // Skip '['.

  memset(&lexer->current.value.char_class, 0,
         sizeof(lexer->current.value.char_class));
  uint8_t* bitmap = lexer->current.value.char_class.bitmap;
  uint16_t* pseudo_mask = &lexer->current.value.char_class.pseudo_mask;
  iree_tokenizer_regex_codepoint_range_t* ranges =
      lexer->current.value.char_class.ranges;
  uint8_t* range_count = &lexer->current.value.char_class.range_count;

  // Check for negation.
  bool negated = false;
  if (iree_tokenizer_regex_lexer_peek_char(lexer) == '^') {
    negated = true;
    iree_tokenizer_regex_lexer_advance_char(lexer);
  }

  // Track if we've seen any content (for validation).
  bool has_content = false;
  // Track pending codepoint for range handling. -1 = none, 0-127 = ASCII (goes
  // in bitmap), 128+ = Unicode codepoint (gets pseudo_mask bit based on script
  // or category).
  int32_t prev_codepoint = -1;
  bool in_range = false;

  while (true) {
    char c = iree_tokenizer_regex_lexer_peek_char(lexer);

    if (c == '\0') {
      iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                           "unterminated character class");
      return false;
    }

    if (c == ']' && has_content) {
      iree_tokenizer_regex_lexer_advance_char(lexer);
      break;
    }

    // Handle escape sequences inside character class.
    if (c == '\\') {
      iree_tokenizer_regex_lexer_advance_char(lexer);
      c = iree_tokenizer_regex_lexer_peek_char(lexer);
      if (c == '\0') {
        iree_tokenizer_regex_lexer_set_error(
            lexer, start_pos, "trailing backslash in char class");
        return false;
      }
      iree_tokenizer_regex_lexer_advance_char(lexer);

      int escape_char = -1;
      switch (c) {
        case 'n':
          escape_char = '\n';
          break;
        case 'r':
          escape_char = '\r';
          break;
        case 't':
          escape_char = '\t';
          break;
        case 'f':
          escape_char = '\f';
          break;
        case 'v':
          escape_char = '\v';
          break;
        case '\\':
        case '-':
        case ']':
        case '[':
        case '^':
          escape_char = c;
          break;

          // Shorthand classes in character class.
          // NOTE: These reset prev_codepoint, so we must flush any pending
          // character.
#define FLUSH_PREV_CODEPOINT()                                          \
  do {                                                                  \
    if (prev_codepoint >= 0) {                                          \
      if (!iree_tokenizer_regex_char_class_add_codepoint(               \
              bitmap, ranges, range_count, (uint32_t)prev_codepoint)) { \
        iree_tokenizer_regex_lexer_set_error(                           \
            lexer, start_pos,                                           \
            "character class exceeds 4 Unicode ranges; use \\p{L} for " \
            "broad matching");                                          \
        return false;                                                   \
      }                                                                 \
    }                                                                   \
  } while (0)
        case 'd':
          FLUSH_PREV_CODEPOINT();
          iree_tokenizer_regex_bitmap_add_shorthand(
              bitmap, pseudo_mask, IREE_TOKENIZER_REGEX_SHORTHAND_d);
          has_content = true;
          prev_codepoint = -1;
          in_range = false;
          continue;
        case 'D':
          FLUSH_PREV_CODEPOINT();
          iree_tokenizer_regex_bitmap_add_shorthand(
              bitmap, pseudo_mask, IREE_TOKENIZER_REGEX_SHORTHAND_D);
          has_content = true;
          prev_codepoint = -1;
          in_range = false;
          continue;
        case 'w':
          FLUSH_PREV_CODEPOINT();
          iree_tokenizer_regex_bitmap_add_shorthand(
              bitmap, pseudo_mask, IREE_TOKENIZER_REGEX_SHORTHAND_w);
          has_content = true;
          prev_codepoint = -1;
          in_range = false;
          continue;
        case 'W':
          FLUSH_PREV_CODEPOINT();
          iree_tokenizer_regex_bitmap_add_shorthand(
              bitmap, pseudo_mask, IREE_TOKENIZER_REGEX_SHORTHAND_W);
          has_content = true;
          prev_codepoint = -1;
          in_range = false;
          continue;
        case 's':
          FLUSH_PREV_CODEPOINT();
          iree_tokenizer_regex_bitmap_add_shorthand(
              bitmap, pseudo_mask, IREE_TOKENIZER_REGEX_SHORTHAND_s);
          has_content = true;
          prev_codepoint = -1;
          in_range = false;
          continue;
        case 'S':
          FLUSH_PREV_CODEPOINT();
          iree_tokenizer_regex_bitmap_add_shorthand(
              bitmap, pseudo_mask, IREE_TOKENIZER_REGEX_SHORTHAND_S);
          has_content = true;
          prev_codepoint = -1;
          in_range = false;
          continue;

        // Unicode property in character class.
        case 'p': {
          if (iree_tokenizer_regex_lexer_peek_char(lexer) != '{') {
            iree_tokenizer_regex_lexer_set_error(lexer, lexer->position - 2,
                                                 "expected '{' after \\p");
            return false;
          }
          iree_tokenizer_regex_lexer_advance_char(lexer);  // Skip '{'.
          iree_host_size_t prop_start = lexer->position;
          while (iree_tokenizer_regex_lexer_peek_char(lexer) != '}' &&
                 iree_tokenizer_regex_lexer_peek_char(lexer) != '\0') {
            iree_tokenizer_regex_lexer_advance_char(lexer);
          }
          if (iree_tokenizer_regex_lexer_peek_char(lexer) != '}') {
            iree_tokenizer_regex_lexer_set_error(lexer, prop_start,
                                                 "unterminated \\p{...}");
            return false;
          }
          iree_host_size_t prop_length = lexer->position - prop_start;
          const char* prop = lexer->input.data + prop_start;
          iree_tokenizer_regex_lexer_advance_char(lexer);  // Skip '}'.

          // Validate property name length: 1-char (L, N, P, M, S, Z, C) or
          // 2-char subcategory (Lu, Nd, etc.). Reject 0-char or 3+ char names.
          if (prop_length != 1 && prop_length != 2) {
            iree_tokenizer_regex_lexer_set_error(lexer, prop_start,
                                                 "unknown Unicode property");
            return false;
          }
          uint8_t pseudo = 0;
          switch (prop[0]) {
            case 'L':
              pseudo = IREE_TOKENIZER_REGEX_PSEUDO_LETTER;
              break;
            case 'N':
              pseudo = IREE_TOKENIZER_REGEX_PSEUDO_NUMBER;
              break;
            case 'P':
              pseudo = IREE_TOKENIZER_REGEX_PSEUDO_PUNCT;
              break;
            case 'M':
              pseudo = IREE_TOKENIZER_REGEX_PSEUDO_MARK;
              break;
            case 'S':
              pseudo = IREE_TOKENIZER_REGEX_PSEUDO_SYMBOL;
              break;
            case 'Z':
              pseudo = IREE_TOKENIZER_REGEX_PSEUDO_SEPARATOR;
              break;
            case 'C':
              pseudo = IREE_TOKENIZER_REGEX_PSEUDO_OTHER;
              break;
            default:
              iree_tokenizer_regex_lexer_set_error(lexer, prop_start,
                                                   "unknown Unicode property");
              return false;
          }
          // Add ASCII characters matching this category using the Unicode lib.
          for (int cc = 0; cc < 128; ++cc) {
            bool matches = false;
            switch (pseudo) {
              case IREE_TOKENIZER_REGEX_PSEUDO_LETTER:
                matches = iree_unicode_is_letter((uint32_t)cc);
                break;
              case IREE_TOKENIZER_REGEX_PSEUDO_NUMBER:
                matches = iree_unicode_is_number((uint32_t)cc);
                break;
              case IREE_TOKENIZER_REGEX_PSEUDO_PUNCT:
                matches = iree_unicode_is_punctuation((uint32_t)cc);
                break;
              case IREE_TOKENIZER_REGEX_PSEUDO_MARK:
                matches = iree_unicode_is_mark((uint32_t)cc);
                break;
              case IREE_TOKENIZER_REGEX_PSEUDO_SYMBOL:
                matches = iree_unicode_is_symbol((uint32_t)cc);
                break;
              case IREE_TOKENIZER_REGEX_PSEUDO_SEPARATOR:
                matches = iree_unicode_is_separator((uint32_t)cc);
                break;
              case IREE_TOKENIZER_REGEX_PSEUDO_OTHER:
                matches = iree_unicode_is_other((uint32_t)cc);
                break;
            }
            if (matches) {
              BITMAP_SET(bitmap, cc);
            }
          }
          *pseudo_mask |= (1u << (pseudo - 0x80));
          // Flush any pending character before resetting prev_codepoint.
          FLUSH_PREV_CODEPOINT();
          has_content = true;
          prev_codepoint = -1;
          in_range = false;
          continue;
        }

        // \uXXXX Unicode escape in character class.
        case 'u': {
          iree_host_size_t escape_start = lexer->position - 2;
          int codepoint = iree_tokenizer_regex_lexer_parse_hex_digits(lexer, 4);
          if (codepoint < 0) {
            iree_tokenizer_regex_lexer_set_error(
                lexer, escape_start,
                "invalid \\u escape: expected 4 hex digits");
            return false;
          }
          if (codepoint > IREE_UNICODE_MAX_CODEPOINT) {
            iree_tokenizer_regex_lexer_set_error(lexer, escape_start,
                                                 "invalid Unicode codepoint");
            return false;
          }

          // ASCII codepoints can be added to bitmap directly.
          if (codepoint < 0x80) {
            escape_char = codepoint;
            break;
          }

          // Non-ASCII: use exact codepoint range for precise matching.
          // This ensures \u4E2D in [xyz\u4E2D] matches only 中, not all CJK.
          FLUSH_PREV_CODEPOINT();
          // Insert current codepoint as single-element range.
          if (!iree_tokenizer_regex_char_class_add_codepoint(
                  bitmap, ranges, range_count, (uint32_t)codepoint)) {
            iree_tokenizer_regex_lexer_set_error(
                lexer, escape_start,
                "character class exceeds 4 Unicode ranges; use \\p{L} for "
                "broad matching");
            return false;
          }
          has_content = true;
          prev_codepoint = -1;
          in_range = false;
          continue;
        }

        default:
          // Unknown escape - reject with error for strict validation.
          iree_tokenizer_regex_lexer_set_error(
              lexer, lexer->position - 2,
              "unknown escape sequence in character class");
          return false;
      }

      if (in_range) {
        // Complete the range.
        if (prev_codepoint < 0 || escape_char < prev_codepoint) {
          iree_tokenizer_regex_lexer_set_error(lexer, lexer->position - 2,
                                               "invalid range");
          return false;
        }
        for (int ch = prev_codepoint; ch <= escape_char; ++ch) {
          BITMAP_SET(bitmap, ch);
        }
        in_range = false;
        prev_codepoint = -1;
      } else {
        // Add the previous character before setting the new one.
        // This matches the behavior of regular character handling.
        FLUSH_PREV_CODEPOINT();
        prev_codepoint = escape_char;
      }
      has_content = true;
      continue;
    }

    // Handle '-' for ranges.
    if (c == '-' && prev_codepoint >= 0 && !in_range) {
      // Peek ahead to see if this is a range or literal '-'.
      char next = iree_tokenizer_regex_lexer_peek_char_at(lexer, 1);
      if (next != ']' && next != '\0') {
        in_range = true;
        iree_tokenizer_regex_lexer_advance_char(lexer);
        continue;
      }
    }

    // Regular character - may be ASCII or UTF-8 multi-byte sequence.
    uint32_t current_codepoint;
    if ((uint8_t)c < 0x80) {
      // ASCII: single byte.
      current_codepoint = (uint8_t)c;
      iree_tokenizer_regex_lexer_advance_char(lexer);
    } else {
      // UTF-8 multi-byte sequence: decode the full codepoint.
      // Create a string_view from current position to end for decoding.
      iree_string_view_t remaining = {
          .data = lexer->input.data + lexer->position,
          .size = lexer->input.size - lexer->position,
      };
      iree_host_size_t decode_pos = 0;
      current_codepoint = iree_unicode_utf8_decode(remaining, &decode_pos);
      // Check for decode failure (prevents infinite loop on invalid UTF-8).
      if (decode_pos == 0) {
        iree_tokenizer_regex_lexer_set_error(
            lexer, lexer->position, "invalid UTF-8 in character class");
        return false;
      }
      // Advance by the number of bytes consumed.
      for (iree_host_size_t i = 0; i < decode_pos; ++i) {
        iree_tokenizer_regex_lexer_advance_char(lexer);
      }
    }

    if (in_range) {
      // Complete the range.
      if (prev_codepoint < 0 || current_codepoint < (uint32_t)prev_codepoint) {
        iree_tokenizer_regex_lexer_set_error(lexer, lexer->position - 1,
                                             "invalid range");
        return false;
      }

      // Determine range handling based on whether endpoints are ASCII or
      // Unicode.
      bool prev_is_ascii = (uint32_t)prev_codepoint < 0x80;
      bool curr_is_ascii = current_codepoint < 0x80;

      if (prev_is_ascii && curr_is_ascii) {
        // ASCII range: fill bitmap.
        for (uint32_t ch = (uint32_t)prev_codepoint; ch <= current_codepoint;
             ++ch) {
          BITMAP_SET(bitmap, ch);
        }
      } else if (!prev_is_ascii && !curr_is_ascii) {
        // Unicode range: store as exact codepoint range for precise matching.
        // This allows patterns like [一-龥] to match only U+4E00-U+9FA5 instead
        // of over-approximating with the entire CJK block.
        if (!iree_tokenizer_regex_insert_sorted_range(ranges, range_count,
                                                      (uint32_t)prev_codepoint,
                                                      current_codepoint)) {
          iree_tokenizer_regex_lexer_set_error(
              lexer, lexer->position - 1,
              "character class exceeds 4 Unicode ranges; use \\p{L} for broad "
              "matching");
          return false;
        }
      } else {
        // Mixed ASCII/Unicode range - not supported.
        iree_tokenizer_regex_lexer_set_error(
            lexer, lexer->position - 1,
            "range between ASCII and Unicode characters not supported");
        return false;
      }
      in_range = false;
      prev_codepoint = -1;
    } else {
      // Just add this character.
      FLUSH_PREV_CODEPOINT();
      prev_codepoint = (int32_t)current_codepoint;
    }
    has_content = true;
  }

  // Don't forget the last character if not in a range.
  if (prev_codepoint >= 0 && !in_range) {
    if (!iree_tokenizer_regex_char_class_add_codepoint(
            bitmap, ranges, range_count, (uint32_t)prev_codepoint)) {
      iree_tokenizer_regex_lexer_set_error(
          lexer, start_pos,
          "character class exceeds 4 Unicode ranges; use \\p{L} for "
          "broad matching");
      return false;
    }
  }
#undef FLUSH_PREV_CODEPOINT

  // Apply negation.
  if (negated) {
    for (int i = 0; i < 32; ++i) {
      bitmap[i] = ~bitmap[i];
    }
    *pseudo_mask = ~(*pseudo_mask);
    lexer->current.value.char_class.negated = true;
  }

  lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_CHAR_CLASS;
  lexer->current.position = start_pos;
  lexer->current.length = lexer->position - start_pos;
  return true;
}

//===----------------------------------------------------------------------===//
// Quantifier Handling
//===----------------------------------------------------------------------===//

// Parses a quantifier {n}, {n,}, {n,m}.
// Assumes position is at the opening '{'.
static bool iree_tokenizer_regex_lexer_parse_quantifier(
    iree_tokenizer_regex_lexer_t* lexer) {
  iree_host_size_t start_pos = lexer->position;
  iree_tokenizer_regex_lexer_advance_char(lexer);  // Skip '{'.

  // Parse min - require at least one digit.
  uint32_t min = 0;
  bool has_min_digit = false;
  while (isdigit(iree_tokenizer_regex_lexer_peek_char(lexer))) {
    has_min_digit = true;
    min = min * 10 + (iree_tokenizer_regex_lexer_peek_char(lexer) - '0');
    if (min >= UINT16_MAX) {
      // UINT16_MAX (65535) is reserved as the "unbounded" marker.
      iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                           "quantifier min too large");
      return false;
    }
    iree_tokenizer_regex_lexer_advance_char(lexer);
  }

  // Validate that min was provided (reject {} and {,N}).
  char c = iree_tokenizer_regex_lexer_peek_char(lexer);
  if (!has_min_digit) {
    if (c == '}') {
      iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                           "empty quantifier {}");
      return false;
    } else if (c == ',') {
      iree_tokenizer_regex_lexer_set_error(
          lexer, start_pos, "quantifier missing count before comma");
      return false;
    }
    // Other chars will be caught by "expected '}'" check below.
  }

  uint32_t max = min;

  if (c == ',') {
    iree_tokenizer_regex_lexer_advance_char(lexer);
    c = iree_tokenizer_regex_lexer_peek_char(lexer);
    if (c == '}') {
      // {n,} - unbounded.
      max = UINT16_MAX;
    } else {
      // {n,m}
      max = 0;
      while (isdigit(iree_tokenizer_regex_lexer_peek_char(lexer))) {
        max = max * 10 + (iree_tokenizer_regex_lexer_peek_char(lexer) - '0');
        if (max >= UINT16_MAX) {
          // UINT16_MAX (65535) is reserved as the "unbounded" marker.
          iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                               "quantifier max too large");
          return false;
        }
        iree_tokenizer_regex_lexer_advance_char(lexer);
      }
      if (max < min) {
        iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                             "quantifier max < min");
        return false;
      }
    }
  }

  if (iree_tokenizer_regex_lexer_peek_char(lexer) != '}') {
    iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                         "expected '}' in quantifier");
    return false;
  }
  iree_tokenizer_regex_lexer_advance_char(lexer);

  // Check for possessive ({n,m}+) or lazy ({n,m}?) modifiers.
  // DFA-based engines are inherently greedy/possessive, so we accept these
  // syntaxes but they have no effect on matching behavior.
  char modifier = iree_tokenizer_regex_lexer_peek_char(lexer);
  if (modifier == '+' || modifier == '?') {
    iree_tokenizer_regex_lexer_advance_char(lexer);
  }

  lexer->current.type = IREE_TOKENIZER_REGEX_TOKEN_QUANTIFIER;
  lexer->current.position = start_pos;
  lexer->current.length = lexer->position - start_pos;
  lexer->current.value.quantifier.min = (uint16_t)min;
  lexer->current.value.quantifier.max = (uint16_t)max;
  return true;
}

//===----------------------------------------------------------------------===//
// Main Scanner
//===----------------------------------------------------------------------===//

static void iree_tokenizer_regex_lexer_scan(
    iree_tokenizer_regex_lexer_t* lexer) {
  iree_host_size_t start_pos = lexer->position;
  char c = iree_tokenizer_regex_lexer_peek_char(lexer);

  if (c == '\0') {
    iree_tokenizer_regex_lexer_set_simple(lexer, IREE_TOKENIZER_REGEX_TOKEN_EOF,
                                          start_pos, 0);
    return;
  }

  iree_tokenizer_regex_lexer_advance_char(lexer);

  switch (c) {
    // Simple meta-characters.
    case '.':
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_DOT, start_pos, 1);
      return;
    case '^':
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_CARET, start_pos, 1);
      return;
    case '$':
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_DOLLAR, start_pos, 1);
      return;
    case '|':
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_PIPE, start_pos, 1);
      return;
    case '*': {
      // Check for possessive (*+) or lazy (*?) modifiers.
      // DFA-based engines are inherently greedy/possessive, so we accept these
      // syntaxes but they have no effect on matching behavior.
      iree_host_size_t length = 1;
      char next = iree_tokenizer_regex_lexer_peek_char(lexer);
      if (next == '+' || next == '?') {
        iree_tokenizer_regex_lexer_advance_char(lexer);
        length = 2;
      }
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_STAR, start_pos, length);
      return;
    }
    case '+': {
      // Check for possessive (++) or lazy (+?) modifiers.
      iree_host_size_t length = 1;
      char next = iree_tokenizer_regex_lexer_peek_char(lexer);
      if (next == '+' || next == '?') {
        iree_tokenizer_regex_lexer_advance_char(lexer);
        length = 2;
      }
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_PLUS, start_pos, length);
      return;
    }
    case '?': {
      // Check for possessive (?+) or lazy (??) modifiers.
      iree_host_size_t length = 1;
      char next = iree_tokenizer_regex_lexer_peek_char(lexer);
      if (next == '+' || next == '?') {
        iree_tokenizer_regex_lexer_advance_char(lexer);
        length = 2;
      }
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_QUESTION, start_pos, length);
      return;
    }
    case ')':
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_RPAREN, start_pos, 1);
      return;

    // Opening parenthesis - check for special groups.
    case '(': {
      char next = iree_tokenizer_regex_lexer_peek_char(lexer);
      if (next == '?') {
        iree_tokenizer_regex_lexer_advance_char(lexer);
        char spec = iree_tokenizer_regex_lexer_peek_char(lexer);
        if (spec == ':') {
          iree_tokenizer_regex_lexer_advance_char(lexer);
          iree_tokenizer_regex_lexer_set_simple(
              lexer, IREE_TOKENIZER_REGEX_TOKEN_GROUP_NC, start_pos, 3);
          return;
        } else if (spec == '!') {
          iree_tokenizer_regex_lexer_advance_char(lexer);
          iree_tokenizer_regex_lexer_set_simple(
              lexer, IREE_TOKENIZER_REGEX_TOKEN_GROUP_NEG_LA, start_pos, 3);
          return;
        } else if (spec == 'i') {
          iree_tokenizer_regex_lexer_advance_char(lexer);
          if (iree_tokenizer_regex_lexer_peek_char(lexer) == ':') {
            iree_tokenizer_regex_lexer_advance_char(lexer);
            iree_tokenizer_regex_lexer_set_simple(
                lexer, IREE_TOKENIZER_REGEX_TOKEN_GROUP_CASE_I, start_pos, 4);
            return;
          }
          // Just (?i without colon - error.
          iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                               "expected ':' after (?i");
          return;
        } else {
          iree_tokenizer_regex_lexer_set_error(lexer, start_pos,
                                               "unknown group type (?");
          return;
        }
      }
      iree_tokenizer_regex_lexer_set_simple(
          lexer, IREE_TOKENIZER_REGEX_TOKEN_LPAREN, start_pos, 1);
      return;
    }

    // Character class.
    case '[':
      lexer->position = start_pos;  // Reset to include '['.
      iree_tokenizer_regex_lexer_parse_char_class(lexer);
      return;

    // Quantifier.
    case '{':
      lexer->position = start_pos;  // Reset to include '{'.
      iree_tokenizer_regex_lexer_parse_quantifier(lexer);
      return;

    // Escape sequence.
    case '\\':
      iree_tokenizer_regex_lexer_parse_escape(lexer, start_pos);
      return;

    // Regular literal character.
    default:
      iree_tokenizer_regex_lexer_set_literal(lexer, (uint8_t)c, start_pos, 1);
      return;
  }
}
