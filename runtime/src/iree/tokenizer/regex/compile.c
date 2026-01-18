// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/regex/compile.h"

#include "iree/base/internal/arena.h"
#include "iree/tokenizer/regex/internal/dfa.h"
#include "iree/tokenizer/regex/internal/nfa.h"
#include "iree/tokenizer/regex/internal/parser.h"

//===----------------------------------------------------------------------===//
// AST Helpers
//===----------------------------------------------------------------------===//

// Recursively sets case_insensitive=true on all nodes in the AST.
// Used to propagate the global CASE_INSENSITIVE flag to all nodes.
static void iree_tokenizer_regex_ast_propagate_case_insensitive(
    iree_tokenizer_regex_ast_node_t* node) {
  if (!node) return;
  node->case_insensitive = true;

  switch (node->type) {
    case IREE_TOKENIZER_REGEX_AST_EMPTY:
    case IREE_TOKENIZER_REGEX_AST_LITERAL:
    case IREE_TOKENIZER_REGEX_AST_DOT:
    case IREE_TOKENIZER_REGEX_AST_CHAR_CLASS:
    case IREE_TOKENIZER_REGEX_AST_SHORTHAND:
    case IREE_TOKENIZER_REGEX_AST_UNICODE_PROP:
    case IREE_TOKENIZER_REGEX_AST_ANCHOR_START:
    case IREE_TOKENIZER_REGEX_AST_ANCHOR_END:
      break;  // Leaf nodes.

    case IREE_TOKENIZER_REGEX_AST_CONCAT:
    case IREE_TOKENIZER_REGEX_AST_ALTERNATION:
      for (iree_host_size_t i = 0; i < node->data.compound.child_count; ++i) {
        iree_tokenizer_regex_ast_propagate_case_insensitive(
            node->data.compound.children[i]);
      }
      break;

    case IREE_TOKENIZER_REGEX_AST_QUANTIFIER:
      iree_tokenizer_regex_ast_propagate_case_insensitive(
          node->data.quantifier.child);
      break;

    case IREE_TOKENIZER_REGEX_AST_GROUP:
      iree_tokenizer_regex_ast_propagate_case_insensitive(
          node->data.group_child);
      break;

    case IREE_TOKENIZER_REGEX_AST_NEG_LOOKAHEAD:
      iree_tokenizer_regex_ast_propagate_case_insensitive(
          node->data.lookahead_child);
      break;
  }
}

//===----------------------------------------------------------------------===//
// Compilation Pipeline
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_regex_compile(
    iree_string_view_t pattern, iree_tokenizer_regex_compile_flags_t flags,
    iree_allocator_t allocator, uint8_t** out_dfa_data,
    iree_host_size_t* out_dfa_size,
    iree_tokenizer_regex_compile_error_t* out_error) {
  IREE_ASSERT_ARGUMENT(out_dfa_data);
  IREE_ASSERT_ARGUMENT(out_dfa_size);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)pattern.size);

  *out_dfa_data = NULL;
  *out_dfa_size = 0;
  if (out_error) {
    out_error->position = 0;
    out_error->length = 0;
    out_error->message = NULL;
  }

  // Reject empty patterns (would create empty-match-everywhere DFA).
  if (pattern.size == 0) {
    if (out_error) {
      out_error->position = 0;
      out_error->message = "empty pattern not allowed";
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty pattern");
  }

  iree_status_t status = iree_ok_status();

  // Arena for all temporary allocations (AST, NFA, DFA construction).
  // This is freed at the end regardless of success/failure.
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/16384, allocator,
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  // Stage 1: Parse pattern to AST.
  iree_tokenizer_regex_ast_node_t* ast = NULL;
  iree_tokenizer_regex_parse_error_t parse_error = {0};
  status = iree_tokenizer_regex_parse(pattern, &arena, &ast, &parse_error);
  if (!iree_status_is_ok(status)) {
    if (out_error) {
      out_error->position = parse_error.position;
      out_error->length = parse_error.length;
      out_error->message = parse_error.message;
    }
    goto cleanup;
  }

  // Apply global case-insensitive flag if set.
  // Must propagate to all nodes, not just root, since NFA builder reads
  // the flag from each individual node.
  if (flags & IREE_TOKENIZER_REGEX_COMPILE_FLAG_CASE_INSENSITIVE) {
    iree_tokenizer_regex_ast_propagate_case_insensitive(ast);
  }

  // Stage 2: Build NFA from AST (Thompson's construction).
  iree_tokenizer_regex_nfa_t nfa = {0};
  status = iree_tokenizer_regex_nfa_build(ast, &arena, &nfa);
  if (!iree_status_is_ok(status)) {
    if (out_error) {
      out_error->message = "NFA construction failed";
    }
    goto cleanup;
  }

  // Stage 3: Build DFA from NFA (subset construction).
  iree_tokenizer_regex_dfa_build_t dfa = {0};
  status = iree_tokenizer_regex_dfa_build(&nfa, &arena, &dfa);
  if (!iree_status_is_ok(status)) {
    if (out_error) {
      out_error->message = "DFA construction failed (state limit?)";
    }
    goto cleanup;
  }

  // Stage 3.5: Validate DFA doesn't match empty strings.
  //
  // The start state being accepting (without end anchor) means the pattern
  // can match empty strings at every position, which would cause infinite
  // loops in the executor. Patterns like a*, a?, (a|)* fall into this category.
  //
  // Patterns with end anchor like ^.*$ are safe because acceptance is deferred
  // until end of input - no intermediate empty matches occur.
  if (dfa.states[0]->is_accepting && !dfa.states[0]->requires_end_anchor) {
    if (out_error) {
      out_error->position = 0;
      out_error->message = "pattern matches empty string";
    }
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pattern matches empty string (use + instead of * for required match)");
    goto cleanup;
  }

  // Additional check: if start is accepting with end anchor but no transitions,
  // it can ONLY match empty strings (e.g., ^$).
  if (dfa.states[0]->is_accepting && dfa.states[0]->requires_end_anchor) {
    bool has_transition = false;
    for (int byte = 0; byte < 256 && !has_transition; ++byte) {
      if (dfa.states[0]->transitions[byte] != UINT32_MAX) {
        has_transition = true;
      }
    }
    if (!has_transition) {
      if (out_error) {
        out_error->position = 0;
        out_error->message = "pattern can only match empty string";
      }
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "pattern can only match empty strings");
      goto cleanup;
    }
  }

  // Stage 4: Serialize DFA to binary format.
  status = iree_tokenizer_regex_dfa_serialize(&dfa, allocator, out_dfa_data,
                                              out_dfa_size);
  if (!iree_status_is_ok(status)) {
    if (out_error) {
      out_error->message = "DFA serialization failed";
    }
    goto cleanup;
  }

cleanup:
  // Free all temporary allocations.
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_regex_compile_and_load(
    iree_string_view_t pattern, iree_tokenizer_regex_compile_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_regex_dfa_t* out_dfa,
    uint8_t** out_dfa_storage,
    iree_tokenizer_regex_compile_error_t* out_error) {
  IREE_ASSERT_ARGUMENT(out_dfa);
  IREE_ASSERT_ARGUMENT(out_dfa_storage);

  *out_dfa_storage = NULL;
  memset(out_dfa, 0, sizeof(*out_dfa));

  // Compile.
  uint8_t* dfa_data = NULL;
  iree_host_size_t dfa_size = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_compile(
      pattern, flags, allocator, &dfa_data, &dfa_size, out_error));

  // Load.
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(dfa_data, dfa_size), out_dfa);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_regex_compiled_free(dfa_data, allocator);
    return status;
  }

  *out_dfa_storage = dfa_data;
  return iree_ok_status();
}

void iree_tokenizer_regex_compiled_free(uint8_t* dfa_data,
                                        iree_allocator_t allocator) {
  if (dfa_data) {
    iree_allocator_free(allocator, dfa_data);
  }
}
