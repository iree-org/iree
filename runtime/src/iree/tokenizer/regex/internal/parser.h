// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_REGEX_INTERNAL_PARSER_H_
#define IREE_TOKENIZER_REGEX_INTERNAL_PARSER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/tokenizer/regex/internal/ast.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Parser Error
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_regex_parse_error_t {
  iree_host_size_t position;  // Byte offset in pattern.
  iree_host_size_t length;    // Span of problematic text.
  const char* message;        // Static error message.
} iree_tokenizer_regex_parse_error_t;

//===----------------------------------------------------------------------===//
// Parser API
//===----------------------------------------------------------------------===//

// Parses a regex pattern and returns the AST root.
//
// |pattern| is the regex pattern string.
// |arena| is used for all AST node allocations (caller manages lifetime).
// |out_ast| receives the root AST node on success.
// |out_error| receives error details on failure (optional, may be NULL).
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - IREE_STATUS_INVALID_ARGUMENT for syntax errors.
//   - IREE_STATUS_FAILED_PRECONDITION for unsupported features.
iree_status_t iree_tokenizer_regex_parse(
    iree_string_view_t pattern, iree_arena_allocator_t* arena,
    iree_tokenizer_regex_ast_node_t** out_ast,
    iree_tokenizer_regex_parse_error_t* out_error);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_REGEX_INTERNAL_PARSER_H_
