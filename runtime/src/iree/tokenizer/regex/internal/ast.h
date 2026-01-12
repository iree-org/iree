// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_REGEX_INTERNAL_AST_H_
#define IREE_TOKENIZER_REGEX_INTERNAL_AST_H_

#include "iree/base/api.h"
#include "iree/tokenizer/regex/exec.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// AST Node Types
//===----------------------------------------------------------------------===//

typedef enum iree_tokenizer_regex_ast_type_e {
  // Leaf nodes.
  IREE_TOKENIZER_REGEX_AST_EMPTY,         // Empty expression.
  IREE_TOKENIZER_REGEX_AST_LITERAL,       // Single byte literal.
  IREE_TOKENIZER_REGEX_AST_DOT,           // . (any character except newline)
  IREE_TOKENIZER_REGEX_AST_CHAR_CLASS,    // Character class [abc].
  IREE_TOKENIZER_REGEX_AST_SHORTHAND,     // \d, \w, \s, etc.
  IREE_TOKENIZER_REGEX_AST_UNICODE_PROP,  // \p{L}, \p{N}, etc.
  IREE_TOKENIZER_REGEX_AST_ANCHOR_START,  // ^
  IREE_TOKENIZER_REGEX_AST_ANCHOR_END,    // $

  // Compound nodes.
  IREE_TOKENIZER_REGEX_AST_CONCAT,         // AB (sequence).
  IREE_TOKENIZER_REGEX_AST_ALTERNATION,    // A|B.
  IREE_TOKENIZER_REGEX_AST_QUANTIFIER,     // A*, A+, A?, A{n,m}.
  IREE_TOKENIZER_REGEX_AST_GROUP,          // (?:A) or (A).
  IREE_TOKENIZER_REGEX_AST_NEG_LOOKAHEAD,  // (?!A).
} iree_tokenizer_regex_ast_type_t;

//===----------------------------------------------------------------------===//
// AST Node Structure
//===----------------------------------------------------------------------===//

// Forward declaration for recursive references.
struct iree_tokenizer_regex_ast_node_t;

typedef struct iree_tokenizer_regex_ast_node_t {
  iree_tokenizer_regex_ast_type_t type;

  // Source position for error reporting.
  iree_host_size_t source_position;

  // Flags.
  bool case_insensitive;  // Set when inside (?i:...).

  union {
    // AST_LITERAL: single byte.
    uint8_t literal;

    // AST_DOT: no extra data.

    // AST_CHAR_CLASS: bitmap + pseudo-byte mask + exact codepoint ranges.
    struct {
      uint8_t bitmap[32];    // 256-bit bitmap for bytes 0-255.
      uint16_t pseudo_mask;  // Mask for pseudo-bytes 0x80-0x87.
      uint8_t range_count;   // Number of exact codepoint ranges (0-4).
      bool negated;
      iree_tokenizer_regex_codepoint_range_t
          ranges[IREE_TOKENIZER_REGEX_MAX_CHAR_CLASS_RANGES];
    } char_class;

    // AST_SHORTHAND: which shorthand class.
    uint8_t shorthand;  // iree_tokenizer_regex_shorthand_t

    // AST_UNICODE_PROP: pseudo-byte value.
    uint8_t unicode_pseudo_byte;

    // AST_CONCAT, AST_ALTERNATION: list of children.
    struct {
      struct iree_tokenizer_regex_ast_node_t** children;
      iree_host_size_t child_count;
      iree_host_size_t child_capacity;
    } compound;

    // AST_QUANTIFIER: child + bounds.
    struct {
      struct iree_tokenizer_regex_ast_node_t* child;
      uint16_t min;
      uint16_t max;  // UINT16_MAX = unbounded.
      bool greedy;   // Always true for now.
    } quantifier;

    // AST_GROUP: child node.
    struct iree_tokenizer_regex_ast_node_t* group_child;

    // AST_NEG_LOOKAHEAD: the lookahead expression.
    // Restricted to single char/shorthand/class.
    struct iree_tokenizer_regex_ast_node_t* lookahead_child;
  } data;
} iree_tokenizer_regex_ast_node_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_REGEX_INTERNAL_AST_H_
