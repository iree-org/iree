// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_REGEX_INTERNAL_LEXER_H_
#define IREE_TOKENIZER_REGEX_INTERNAL_LEXER_H_

#include "iree/base/api.h"
#include "iree/tokenizer/regex/exec.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Token Types
//===----------------------------------------------------------------------===//

typedef enum iree_tokenizer_regex_token_type_e {
  // Literal byte value (after escape processing).
  IREE_TOKENIZER_REGEX_TOKEN_LITERAL,

  // Meta-characters.
  IREE_TOKENIZER_REGEX_TOKEN_DOT,       // .
  IREE_TOKENIZER_REGEX_TOKEN_CARET,     // ^
  IREE_TOKENIZER_REGEX_TOKEN_DOLLAR,    // $
  IREE_TOKENIZER_REGEX_TOKEN_PIPE,      // |
  IREE_TOKENIZER_REGEX_TOKEN_STAR,      // *
  IREE_TOKENIZER_REGEX_TOKEN_PLUS,      // +
  IREE_TOKENIZER_REGEX_TOKEN_QUESTION,  // ?

  // Grouping.
  IREE_TOKENIZER_REGEX_TOKEN_LPAREN,  // (
  IREE_TOKENIZER_REGEX_TOKEN_RPAREN,  // )
  IREE_TOKENIZER_REGEX_TOKEN_LBRACE,  // {
  IREE_TOKENIZER_REGEX_TOKEN_RBRACE,  // }

  // Special group types (detected by lexer).
  IREE_TOKENIZER_REGEX_TOKEN_GROUP_NC,      // (?:
  IREE_TOKENIZER_REGEX_TOKEN_GROUP_CASE_I,  // (?i:
  IREE_TOKENIZER_REGEX_TOKEN_GROUP_NEG_LA,  // (?!

  // Character class (parsed to bitmap in lexer).
  IREE_TOKENIZER_REGEX_TOKEN_CHAR_CLASS,

  // Shorthand classes (\d, \D, \w, \W, \s, \S).
  IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND,

  // Unicode property (\p{L}, \p{N}, etc.).
  IREE_TOKENIZER_REGEX_TOKEN_UNICODE_PROP,

  // Quantifier with bounds {n}, {n,}, {n,m}.
  IREE_TOKENIZER_REGEX_TOKEN_QUANTIFIER,

  // End of input.
  IREE_TOKENIZER_REGEX_TOKEN_EOF,

  // Lexer error (message in error field).
  IREE_TOKENIZER_REGEX_TOKEN_ERROR,
} iree_tokenizer_regex_token_type_t;

//===----------------------------------------------------------------------===//
// Token Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_regex_token_t {
  iree_tokenizer_regex_token_type_t type;

  // Position in source pattern (for error reporting).
  iree_host_size_t position;
  iree_host_size_t length;

  union {
    // For TOKEN_LITERAL: the literal byte value.
    uint8_t literal;

    // For TOKEN_SHORTHAND: which shorthand class.
    iree_tokenizer_regex_shorthand_t shorthand;

    // For TOKEN_UNICODE_PROP: the pseudo-byte value (0x80-0x87).
    uint8_t unicode_pseudo_byte;

    // For TOKEN_QUANTIFIER: min and max bounds.
    struct {
      uint16_t min;
      uint16_t max;  // UINT16_MAX for unbounded ({n,}).
    } quantifier;

    // For TOKEN_CHAR_CLASS: bitmap + pseudo-byte mask + exact codepoint ranges.
    struct {
      uint8_t bitmap[32];    // Bitmap for ASCII bytes 0-255.
      uint16_t pseudo_mask;  // Mask for pseudo-bytes 0x80-0x87.
      uint8_t range_count;   // Number of exact codepoint ranges (0-4).
      bool negated;
      iree_tokenizer_regex_codepoint_range_t
          ranges[IREE_TOKENIZER_REGEX_MAX_CHAR_CLASS_RANGES];
    } char_class;

    // For TOKEN_ERROR: error message (static string).
    const char* error_message;
  } value;
} iree_tokenizer_regex_token_t;

//===----------------------------------------------------------------------===//
// Lexer State
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_regex_lexer_t {
  iree_string_view_t input;
  iree_host_size_t position;
  iree_tokenizer_regex_token_t current;
  bool has_peeked;
} iree_tokenizer_regex_lexer_t;

//===----------------------------------------------------------------------===//
// Lexer API
//===----------------------------------------------------------------------===//

// Initializes the lexer with the given pattern.
void iree_tokenizer_regex_lexer_initialize(iree_tokenizer_regex_lexer_t* lexer,
                                           iree_string_view_t pattern);

// Peeks at the current token without consuming it.
// Returns pointer to internal token; valid until next peek/advance.
const iree_tokenizer_regex_token_t* iree_tokenizer_regex_lexer_peek(
    iree_tokenizer_regex_lexer_t* lexer);

// Advances past the current token.
void iree_tokenizer_regex_lexer_advance(iree_tokenizer_regex_lexer_t* lexer);

// Returns the current position in the input for error reporting.
iree_host_size_t iree_tokenizer_regex_lexer_position(
    const iree_tokenizer_regex_lexer_t* lexer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_REGEX_INTERNAL_LEXER_H_
