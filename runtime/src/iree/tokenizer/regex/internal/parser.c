// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/regex/internal/parser.h"

#include <string.h>

#include "iree/tokenizer/regex/internal/lexer.h"

//===----------------------------------------------------------------------===//
// Parser State
//===----------------------------------------------------------------------===//

// Maximum nesting depth for groups to prevent stack overflow.
// Deeply nested patterns like ((((a)))) repeated many times would exhaust
// the call stack during recursive descent parsing.
#define IREE_TOKENIZER_REGEX_MAX_NESTING_DEPTH 100

typedef struct iree_tokenizer_regex_parser_t {
  iree_tokenizer_regex_lexer_t lexer;
  iree_arena_allocator_t* arena;
  iree_tokenizer_regex_parse_error_t
      error;              // User-facing: position in pattern.
  iree_status_t status;   // Non-OK on any error (owns allocation).
  bool case_insensitive;  // Currently inside (?i:...).
  uint32_t depth;         // Current nesting depth.
} iree_tokenizer_regex_parser_t;

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

static iree_tokenizer_regex_ast_node_t*
iree_tokenizer_regex_parser_parse_alternation(
    iree_tokenizer_regex_parser_t* parser);

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Sets parser error from a syntax issue (populates user-facing error info).
// Only captures the first error; subsequent errors are ignored.
static void iree_tokenizer_regex_parser_set_error(
    iree_tokenizer_regex_parser_t* parser, iree_host_size_t position,
    const char* message) {
  if (iree_status_is_ok(parser->status)) {
    parser->error.position = position;
    parser->error.length = 1;
    parser->error.message = message;
    parser->status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "parse error at position %zu: %s", position, message);
  }
}

// Sets parser error from an internal failure (e.g., allocation).
// Transfers ownership of the status to the parser.
// Only captures the first error; subsequent errors are freed.
static void iree_tokenizer_regex_parser_set_status(
    iree_tokenizer_regex_parser_t* parser, iree_status_t status) {
  if (iree_status_is_ok(parser->status)) {
    parser->status = status;
  } else {
    iree_status_ignore(status);
  }
}

// Allocates an AST node from the arena.
static iree_tokenizer_regex_ast_node_t*
iree_tokenizer_regex_parser_node_allocate(iree_tokenizer_regex_parser_t* parser,
                                          iree_tokenizer_regex_ast_type_t type,
                                          iree_host_size_t position) {
  iree_tokenizer_regex_ast_node_t* node = NULL;
  iree_status_t status =
      iree_arena_allocate(parser->arena, sizeof(*node), (void**)&node);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_regex_parser_set_status(parser, status);
    return NULL;
  }
  memset(node, 0, sizeof(*node));
  node->type = type;
  node->source_position = position;
  node->case_insensitive = parser->case_insensitive;
  return node;
}

// Peeks at current token.
static const iree_tokenizer_regex_token_t* iree_tokenizer_regex_parser_peek(
    iree_tokenizer_regex_parser_t* parser) {
  return iree_tokenizer_regex_lexer_peek(&parser->lexer);
}

// Advances past current token.
static void iree_tokenizer_regex_parser_advance(
    iree_tokenizer_regex_parser_t* parser) {
  iree_tokenizer_regex_lexer_advance(&parser->lexer);
}

// Checks if current token is of given type.
static bool iree_tokenizer_regex_parser_check(
    iree_tokenizer_regex_parser_t* parser,
    iree_tokenizer_regex_token_type_t type) {
  return iree_tokenizer_regex_parser_peek(parser)->type == type;
}

// Consumes token if it matches, returns true if consumed.
static bool iree_tokenizer_regex_parser_match(
    iree_tokenizer_regex_parser_t* parser,
    iree_tokenizer_regex_token_type_t type) {
  if (iree_tokenizer_regex_parser_check(parser, type)) {
    iree_tokenizer_regex_parser_advance(parser);
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Compound Node Helpers
//===----------------------------------------------------------------------===//

// Adds a child to a compound node (CONCAT or ALTERNATION).
static bool iree_tokenizer_regex_parser_add_child(
    iree_tokenizer_regex_parser_t* parser,
    iree_tokenizer_regex_ast_node_t* parent,
    iree_tokenizer_regex_ast_node_t* child) {
  if (!child) return false;

  iree_host_size_t count = parent->data.compound.child_count;
  iree_host_size_t capacity = parent->data.compound.child_capacity;

  if (count >= capacity) {
    // Grow capacity.
    iree_host_size_t new_capacity = capacity == 0 ? 4 : capacity * 2;
    iree_tokenizer_regex_ast_node_t** new_children = NULL;
    iree_status_t status = iree_arena_allocate(
        parser->arena, new_capacity * sizeof(iree_tokenizer_regex_ast_node_t*),
        (void**)&new_children);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_regex_parser_set_status(parser, status);
      return false;
    }
    if (parent->data.compound.children) {
      memcpy(new_children, parent->data.compound.children,
             count * sizeof(iree_tokenizer_regex_ast_node_t*));
    }
    parent->data.compound.children = new_children;
    parent->data.compound.child_capacity = new_capacity;
  }

  parent->data.compound.children[count] = child;
  parent->data.compound.child_count = count + 1;
  return true;
}

//===----------------------------------------------------------------------===//
// Grammar: atom
//===----------------------------------------------------------------------===//

// atom -> literal | charclass | shorthand | unicode | '.' | '^' | '$' | group

static iree_tokenizer_regex_ast_node_t* iree_tokenizer_regex_parser_parse_atom(
    iree_tokenizer_regex_parser_t* parser) {
  const iree_tokenizer_regex_token_t* tok =
      iree_tokenizer_regex_parser_peek(parser);

  switch (tok->type) {
    case IREE_TOKENIZER_REGEX_TOKEN_LITERAL: {
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_LITERAL, tok->position);
      if (!node) return NULL;
      node->data.literal = tok->value.literal;
      iree_tokenizer_regex_parser_advance(parser);
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_DOT: {
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_DOT, tok->position);
      iree_tokenizer_regex_parser_advance(parser);
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_CARET: {
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_ANCHOR_START, tok->position);
      iree_tokenizer_regex_parser_advance(parser);
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_DOLLAR: {
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_ANCHOR_END, tok->position);
      iree_tokenizer_regex_parser_advance(parser);
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_CHAR_CLASS: {
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_CHAR_CLASS, tok->position);
      if (!node) return NULL;
      memcpy(node->data.char_class.bitmap, tok->value.char_class.bitmap, 32);
      node->data.char_class.pseudo_mask = tok->value.char_class.pseudo_mask;
      node->data.char_class.negated = tok->value.char_class.negated;
      node->data.char_class.range_count = tok->value.char_class.range_count;
      memcpy(node->data.char_class.ranges, tok->value.char_class.ranges,
             sizeof(node->data.char_class.ranges));
      iree_tokenizer_regex_parser_advance(parser);
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND: {
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_SHORTHAND, tok->position);
      if (!node) return NULL;
      node->data.shorthand = (uint8_t)tok->value.shorthand;
      iree_tokenizer_regex_parser_advance(parser);
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_UNICODE_PROP: {
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_UNICODE_PROP, tok->position);
      if (!node) return NULL;
      node->data.unicode_pseudo_byte = tok->value.unicode_pseudo_byte;
      iree_tokenizer_regex_parser_advance(parser);
      return node;
    }

    // Groups.
    case IREE_TOKENIZER_REGEX_TOKEN_LPAREN:
    case IREE_TOKENIZER_REGEX_TOKEN_GROUP_NC: {
      iree_host_size_t position = tok->position;
      // Check depth limit before recursing.
      if (parser->depth >= IREE_TOKENIZER_REGEX_MAX_NESTING_DEPTH) {
        iree_tokenizer_regex_parser_set_error(
            parser, position,
            "maximum nesting depth exceeded (100 levels); simplify pattern");
        return NULL;
      }
      iree_tokenizer_regex_parser_advance(parser);
      ++parser->depth;
      iree_tokenizer_regex_ast_node_t* inner =
          iree_tokenizer_regex_parser_parse_alternation(parser);
      --parser->depth;
      if (!inner) return NULL;
      if (!iree_tokenizer_regex_parser_match(
              parser, IREE_TOKENIZER_REGEX_TOKEN_RPAREN)) {
        iree_tokenizer_regex_parser_set_error(parser, position,
                                              "unbalanced parentheses");
        return NULL;
      }
      // For non-capturing groups, we can just return the inner node.
      // The group wrapper is only needed if we want to track case-insensitive.
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_GROUP, position);
      if (!node) return NULL;
      node->data.group_child = inner;
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_GROUP_CASE_I: {
      iree_host_size_t position = tok->position;
      // Check depth limit before recursing.
      if (parser->depth >= IREE_TOKENIZER_REGEX_MAX_NESTING_DEPTH) {
        iree_tokenizer_regex_parser_set_error(
            parser, position,
            "maximum nesting depth exceeded (100 levels); simplify pattern");
        return NULL;
      }
      iree_tokenizer_regex_parser_advance(parser);
      ++parser->depth;
      bool prev_case_insensitive = parser->case_insensitive;
      parser->case_insensitive = true;
      iree_tokenizer_regex_ast_node_t* inner =
          iree_tokenizer_regex_parser_parse_alternation(parser);
      parser->case_insensitive = prev_case_insensitive;
      --parser->depth;
      if (!inner) return NULL;
      if (!iree_tokenizer_regex_parser_match(
              parser, IREE_TOKENIZER_REGEX_TOKEN_RPAREN)) {
        iree_tokenizer_regex_parser_set_error(parser, position,
                                              "unbalanced parentheses");
        return NULL;
      }
      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_GROUP, position);
      if (!node) return NULL;
      node->data.group_child = inner;
      node->case_insensitive = true;  // Mark the group itself.
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_GROUP_NEG_LA: {
      iree_host_size_t position = tok->position;
      iree_tokenizer_regex_parser_advance(parser);
      // Negative lookahead: must be a simple atom (single
      // char/shorthand/class).
      const iree_tokenizer_regex_token_t* la_tok =
          iree_tokenizer_regex_parser_peek(parser);
      iree_tokenizer_regex_ast_node_t* la_child = NULL;

      switch (la_tok->type) {
        case IREE_TOKENIZER_REGEX_TOKEN_LITERAL:
        case IREE_TOKENIZER_REGEX_TOKEN_CHAR_CLASS:
        case IREE_TOKENIZER_REGEX_TOKEN_SHORTHAND:
        case IREE_TOKENIZER_REGEX_TOKEN_UNICODE_PROP:
        case IREE_TOKENIZER_REGEX_TOKEN_DOT:
          la_child = iree_tokenizer_regex_parser_parse_atom(parser);
          break;
        default:
          iree_tokenizer_regex_parser_set_error(
              parser, la_tok->position,
              "negative lookahead must contain single char/class");
          return NULL;
      }

      if (!la_child) return NULL;
      if (!iree_tokenizer_regex_parser_match(
              parser, IREE_TOKENIZER_REGEX_TOKEN_RPAREN)) {
        iree_tokenizer_regex_parser_set_error(
            parser, position, "unbalanced parentheses in lookahead");
        return NULL;
      }

      iree_tokenizer_regex_ast_node_t* node =
          iree_tokenizer_regex_parser_node_allocate(
              parser, IREE_TOKENIZER_REGEX_AST_NEG_LOOKAHEAD, position);
      if (!node) return NULL;
      node->data.lookahead_child = la_child;
      return node;
    }

    case IREE_TOKENIZER_REGEX_TOKEN_ERROR:
      iree_tokenizer_regex_parser_set_error(parser, tok->position,
                                            tok->value.error_message);
      return NULL;

    default:
      // Not an atom - return NULL to signal end of sequence.
      return NULL;
  }
}

//===----------------------------------------------------------------------===//
// Grammar: quantified
//===----------------------------------------------------------------------===//

// quantified -> atom quantifier?

static iree_tokenizer_regex_ast_node_t*
iree_tokenizer_regex_parser_parse_quantified(
    iree_tokenizer_regex_parser_t* parser) {
  iree_tokenizer_regex_ast_node_t* atom =
      iree_tokenizer_regex_parser_parse_atom(parser);
  if (!atom) return NULL;

  const iree_tokenizer_regex_token_t* tok =
      iree_tokenizer_regex_parser_peek(parser);
  uint16_t min = 0, max = 0;
  bool has_quantifier = false;

  switch (tok->type) {
    case IREE_TOKENIZER_REGEX_TOKEN_STAR:
      min = 0;
      max = UINT16_MAX;
      has_quantifier = true;
      iree_tokenizer_regex_parser_advance(parser);
      break;
    case IREE_TOKENIZER_REGEX_TOKEN_PLUS:
      min = 1;
      max = UINT16_MAX;
      has_quantifier = true;
      iree_tokenizer_regex_parser_advance(parser);
      break;
    case IREE_TOKENIZER_REGEX_TOKEN_QUESTION:
      min = 0;
      max = 1;
      has_quantifier = true;
      iree_tokenizer_regex_parser_advance(parser);
      break;
    case IREE_TOKENIZER_REGEX_TOKEN_QUANTIFIER:
      min = tok->value.quantifier.min;
      max = tok->value.quantifier.max;
      has_quantifier = true;
      iree_tokenizer_regex_parser_advance(parser);
      break;
    default:
      break;
  }

  if (!has_quantifier) {
    return atom;
  }

  // Check for nested quantifiers: (expr+)+ or similar.
  // These cause exponential DFA state explosion and are never needed for
  // tokenizer patterns. Reject with a clear error.
  if (atom->type == IREE_TOKENIZER_REGEX_AST_GROUP) {
    iree_tokenizer_regex_ast_node_t* group_child = atom->data.group_child;
    // Unwrap through single-element CONCAT if present.
    if (group_child && group_child->type == IREE_TOKENIZER_REGEX_AST_CONCAT &&
        group_child->data.compound.child_count == 1) {
      group_child = group_child->data.compound.children[0];
    }
    if (group_child &&
        group_child->type == IREE_TOKENIZER_REGEX_AST_QUANTIFIER) {
      iree_tokenizer_regex_parser_set_error(
          parser, atom->source_position,
          "nested quantifiers like (a+)+ cause exponential state explosion; "
          "rewrite pattern");
      return NULL;
    }
  }

  iree_tokenizer_regex_ast_node_t* node =
      iree_tokenizer_regex_parser_node_allocate(
          parser, IREE_TOKENIZER_REGEX_AST_QUANTIFIER, atom->source_position);
  if (!node) return NULL;
  node->data.quantifier.child = atom;
  node->data.quantifier.min = min;
  node->data.quantifier.max = max;
  node->data.quantifier.greedy = true;
  return node;
}

//===----------------------------------------------------------------------===//
// Grammar: concat
//===----------------------------------------------------------------------===//

// concat -> quantified+

static iree_tokenizer_regex_ast_node_t*
iree_tokenizer_regex_parser_parse_concat(
    iree_tokenizer_regex_parser_t* parser) {
  iree_tokenizer_regex_ast_node_t* first =
      iree_tokenizer_regex_parser_parse_quantified(parser);
  if (!first) {
    // Empty sequence is valid (e.g., in alternation branch).
    return iree_tokenizer_regex_parser_node_allocate(
        parser, IREE_TOKENIZER_REGEX_AST_EMPTY,
        iree_tokenizer_regex_lexer_position(&parser->lexer));
  }

  // Parse additional items. CONCAT node is allocated lazily only when we
  // actually have multiple items - this avoids wasting arena space when
  // the peek below shows a non-terminator but the subsequent parse fails.
  iree_tokenizer_regex_ast_node_t* concat = NULL;
  while (true) {
    const iree_tokenizer_regex_token_t* tok =
        iree_tokenizer_regex_parser_peek(parser);
    if (tok->type == IREE_TOKENIZER_REGEX_TOKEN_EOF ||
        tok->type == IREE_TOKENIZER_REGEX_TOKEN_PIPE ||
        tok->type == IREE_TOKENIZER_REGEX_TOKEN_RPAREN) {
      break;
    }

    iree_tokenizer_regex_ast_node_t* item =
        iree_tokenizer_regex_parser_parse_quantified(parser);
    if (!item) {
      if (!iree_status_is_ok(parser->status)) return NULL;
      break;
    }

    // Lazy allocation: create CONCAT only when we confirm a second item.
    if (!concat) {
      concat = iree_tokenizer_regex_parser_node_allocate(
          parser, IREE_TOKENIZER_REGEX_AST_CONCAT, first->source_position);
      if (!concat) return NULL;
      if (!iree_tokenizer_regex_parser_add_child(parser, concat, first))
        return NULL;
    }
    if (!iree_tokenizer_regex_parser_add_child(parser, concat, item))
      return NULL;
  }

  // Return first directly if no additional items were parsed.
  if (!concat) return first;

  // Validate: lookahead can only appear at the end of a concatenation.
  // Mid-pattern lookahead would be silently ignored by the NFA builder.
  // Exception: end anchors ($) can follow lookahead since they're not content.
  for (iree_host_size_t i = 0; i < concat->data.compound.child_count - 1; ++i) {
    iree_tokenizer_regex_ast_node_t* child = concat->data.compound.children[i];
    if (child->type == IREE_TOKENIZER_REGEX_AST_NEG_LOOKAHEAD) {
      // Check if all remaining children are end anchors.
      bool only_end_anchors_follow = true;
      for (iree_host_size_t j = i + 1; j < concat->data.compound.child_count;
           ++j) {
        if (concat->data.compound.children[j]->type !=
            IREE_TOKENIZER_REGEX_AST_ANCHOR_END) {
          only_end_anchors_follow = false;
          break;
        }
      }
      if (!only_end_anchors_follow) {
        iree_tokenizer_regex_parser_set_error(
            parser, child->source_position,
            "negative lookahead must be at end of pattern or branch");
        return NULL;
      }
    }
  }

  return concat;
}

//===----------------------------------------------------------------------===//
// Grammar: alternation
//===----------------------------------------------------------------------===//

// alternation -> concat ('|' concat)*

static iree_tokenizer_regex_ast_node_t*
iree_tokenizer_regex_parser_parse_alternation(
    iree_tokenizer_regex_parser_t* parser) {
  iree_tokenizer_regex_ast_node_t* first =
      iree_tokenizer_regex_parser_parse_concat(parser);
  if (!first) return NULL;

  if (!iree_tokenizer_regex_parser_check(parser,
                                         IREE_TOKENIZER_REGEX_TOKEN_PIPE)) {
    // No alternation.
    return first;
  }

  // Multiple alternatives - create ALTERNATION node.
  iree_tokenizer_regex_ast_node_t* alt =
      iree_tokenizer_regex_parser_node_allocate(
          parser, IREE_TOKENIZER_REGEX_AST_ALTERNATION, first->source_position);
  if (!alt) return NULL;

  if (!iree_tokenizer_regex_parser_add_child(parser, alt, first)) return NULL;

  while (iree_tokenizer_regex_parser_match(parser,
                                           IREE_TOKENIZER_REGEX_TOKEN_PIPE)) {
    iree_tokenizer_regex_ast_node_t* branch =
        iree_tokenizer_regex_parser_parse_concat(parser);
    if (!branch) {
      if (!iree_status_is_ok(parser->status)) return NULL;
      // Empty branch is valid.
      branch = iree_tokenizer_regex_parser_node_allocate(
          parser, IREE_TOKENIZER_REGEX_AST_EMPTY,
          iree_tokenizer_regex_lexer_position(&parser->lexer));
      if (!branch) return NULL;
    }
    if (!iree_tokenizer_regex_parser_add_child(parser, alt, branch))
      return NULL;
  }

  return alt;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_regex_parse(
    iree_string_view_t pattern, iree_arena_allocator_t* arena,
    iree_tokenizer_regex_ast_node_t** out_ast,
    iree_tokenizer_regex_parse_error_t* out_error) {
  if (!arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "arena is NULL");
  }
  if (!out_ast) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_ast is NULL");
  }
  *out_ast = NULL;

  iree_tokenizer_regex_parser_t parser;
  memset(&parser, 0, sizeof(parser));  // Zeroes status to iree_ok_status().
  iree_tokenizer_regex_lexer_initialize(&parser.lexer, pattern);
  parser.arena = arena;

  iree_tokenizer_regex_ast_node_t* ast =
      iree_tokenizer_regex_parser_parse_alternation(&parser);

  // Check for any error (syntax or internal).
  if (!iree_status_is_ok(parser.status)) {
    if (out_error) {
      *out_error = parser.error;
    }
    return parser.status;  // Transfer ownership to caller.
  }

  if (!ast) {
    // Should not happen if status is OK - indicates a bug in the parser.
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "parser returned NULL without setting error");
  }

  // Verify we consumed all input.
  if (!iree_tokenizer_regex_parser_check(&parser,
                                         IREE_TOKENIZER_REGEX_TOKEN_EOF)) {
    const iree_tokenizer_regex_token_t* tok =
        iree_tokenizer_regex_parser_peek(&parser);
    if (out_error) {
      out_error->position = tok->position;
      out_error->length = tok->length;
      out_error->message = "unexpected token";
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected token at position %zu", tok->position);
  }

  *out_ast = ast;
  return iree_ok_status();
}
