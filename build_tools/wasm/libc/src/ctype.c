// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <ctype.h> for wasm32.
// ASCII-only, no locale. Single lookup table for all classification.
//
// Each byte in the table encodes a bitmask of character properties.
// This gives O(1) classification with a single table load — no branches
// for any of the is* functions.

#include <ctype.h>

// Property bits packed into each table entry.
#define CT_UPPER 0x01
#define CT_LOWER 0x02
#define CT_DIGIT 0x04
#define CT_CNTRL 0x08
#define CT_PUNCT 0x10
#define CT_SPACE 0x20
#define CT_HEX 0x40
#define CT_BLANK 0x80

// Derived masks.
#define CT_ALPHA (CT_UPPER | CT_LOWER)
#define CT_ALNUM (CT_ALPHA | CT_DIGIT)
#define CT_GRAPH (CT_ALNUM | CT_PUNCT)
#define CT_PRINT (CT_GRAPH | CT_BLANK)

// clang-format off
static const unsigned char ctype_table[256] = {
  // 0x00-0x08: control characters.
  CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL,
  CT_CNTRL,
  // 0x09: horizontal tab (control + space + blank).
  CT_CNTRL | CT_SPACE | CT_BLANK,
  // 0x0A-0x0D: LF, VT, FF, CR (control + space).
  CT_CNTRL | CT_SPACE, CT_CNTRL | CT_SPACE, CT_CNTRL | CT_SPACE, CT_CNTRL | CT_SPACE,
  // 0x0E-0x1F: control characters.
  CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL,
  CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL, CT_CNTRL,
  CT_CNTRL, CT_CNTRL,
  // 0x20: space (space + blank, also "printable" via CT_BLANK in the PRINT mask).
  CT_SPACE | CT_BLANK,
  // 0x21-0x2F: ! " # $ % & ' ( ) * + , - . /
  CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT,
  CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT,
  // 0x30-0x39: 0-9 (digit + hex).
  CT_DIGIT | CT_HEX, CT_DIGIT | CT_HEX, CT_DIGIT | CT_HEX, CT_DIGIT | CT_HEX,
  CT_DIGIT | CT_HEX, CT_DIGIT | CT_HEX, CT_DIGIT | CT_HEX, CT_DIGIT | CT_HEX,
  CT_DIGIT | CT_HEX, CT_DIGIT | CT_HEX,
  // 0x3A-0x40: : ; < = > ? @
  CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT,
  // 0x41-0x46: A-F (upper + hex).
  CT_UPPER | CT_HEX, CT_UPPER | CT_HEX, CT_UPPER | CT_HEX,
  CT_UPPER | CT_HEX, CT_UPPER | CT_HEX, CT_UPPER | CT_HEX,
  // 0x47-0x5A: G-Z (upper).
  CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER,
  CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER,
  CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER, CT_UPPER,
  // 0x5B-0x60: [ \ ] ^ _ `
  CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT,
  // 0x61-0x66: a-f (lower + hex).
  CT_LOWER | CT_HEX, CT_LOWER | CT_HEX, CT_LOWER | CT_HEX,
  CT_LOWER | CT_HEX, CT_LOWER | CT_HEX, CT_LOWER | CT_HEX,
  // 0x67-0x7A: g-z (lower).
  CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER,
  CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER,
  CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER, CT_LOWER,
  // 0x7B-0x7E: { | } ~
  CT_PUNCT, CT_PUNCT, CT_PUNCT, CT_PUNCT,
  // 0x7F: DEL (control).
  CT_CNTRL,
  // 0x80-0xFF: high bytes — all zero (non-ASCII is unclassified).
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};
// clang-format on

int isalnum(int c) { return ctype_table[(unsigned char)c] & CT_ALNUM; }
int isalpha(int c) { return ctype_table[(unsigned char)c] & CT_ALPHA; }
int isblank(int c) { return ctype_table[(unsigned char)c] & CT_BLANK; }
int iscntrl(int c) { return ctype_table[(unsigned char)c] & CT_CNTRL; }
int isdigit(int c) { return ctype_table[(unsigned char)c] & CT_DIGIT; }
int isgraph(int c) { return ctype_table[(unsigned char)c] & CT_GRAPH; }
int islower(int c) { return ctype_table[(unsigned char)c] & CT_LOWER; }
int isprint(int c) {
  // Printable = graph + space. Space (0x20) has CT_BLANK but not any graph
  // bits, so we check for the union.
  unsigned char flags = ctype_table[(unsigned char)c];
  return flags & (CT_GRAPH | CT_BLANK);
}
int ispunct(int c) { return ctype_table[(unsigned char)c] & CT_PUNCT; }
int isspace(int c) { return ctype_table[(unsigned char)c] & CT_SPACE; }
int isupper(int c) { return ctype_table[(unsigned char)c] & CT_UPPER; }
int isxdigit(int c) { return ctype_table[(unsigned char)c] & CT_HEX; }

int tolower(int c) {
  return (ctype_table[(unsigned char)c] & CT_UPPER) ? (c | 0x20) : c;
}

int toupper(int c) {
  return (ctype_table[(unsigned char)c] & CT_LOWER) ? (c & ~0x20) : c;
}
