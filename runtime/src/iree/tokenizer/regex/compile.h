// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Regex-to-DFA compiler for tokenizer pre-tokenization patterns.
//
// This compiler transforms regex patterns (like those used in GPT-2, Llama 3,
// and Qwen2 tokenizers) into the binary DFA format defined in exec.h. The
// compilation follows a classic pipeline:
//
//   Pattern string → Lexer → Parser → AST → NFA → DFA → Binary
//
// Supported Features (covers 95%+ of HuggingFace tokenizers):
//   - Literals: a, ab, abc
//   - Any character: .
//   - Character classes: [abc], [a-z], [^abc]
//   - Escapes: \r, \n, \t, \f, \v, \a, \e, \0, \\, \[, etc.
//   - Shorthand classes: \d, \D, \w, \W, \s, \S
//   - Unicode categories: \p{L}, \p{N}, \p{P}, \p{M}, \p{S}, \p{Z}, \p{C}
//     (Two-letter subcategories like Lu, Nd map to their general category)
//   - Quantifiers: *, +, ?, {n}, {n,}, {n,m}
//   - Alternation: a|b|c
//   - Grouping: (...), (?:...)
//   - Case-insensitive: (?i:...)
//   - Anchors: ^, $
//   - Negative lookahead: (?!x), (?!\S) (restrictions below)
//
// Negative Lookahead Restrictions:
//   - Must be at end of alternation branch (not mid-pattern)
//   - Supports: single literal char, shorthand class (\s, \S, \d, etc.)
//   - NOT supported: character class (?![abc]), multi-char patterns
//   Valid:   \s+(?!\S)     - whitespace not followed by non-whitespace
//   Valid:   word(?!s)     - "word" not followed by "s"
//   Invalid: a(?!b)c       - lookahead in middle of pattern
//   Invalid: a(?![bc])     - character class in lookahead
//
// NOT Supported (rare, complex, breaks DFA properties):
//   - Backreferences: \1, \2 (requires backtracking)
//   - Positive lookahead: (?=...) (complex state machine)
//   - Lookbehind: (?<=...), (?<!...) (requires backward scan)
//   - Negated Unicode: \P{...} (use [^\p{...}] instead)
//
// Usage:
//   uint8_t* dfa_data = NULL;
//   iree_host_size_t dfa_size = 0;
//   iree_tokenizer_regex_compile_error_t error = {0};
//   iree_status_t status = iree_tokenizer_regex_compile(
//       iree_make_cstring_view("\\s+(?!\\S)|\\s+"),
//       IREE_TOKENIZER_REGEX_COMPILE_FLAG_NONE,
//       iree_allocator_system(),
//       &dfa_data, &dfa_size, &error);
//   if (!iree_status_is_ok(status)) {
//     // error.position, error.message contain details
//     iree_status_ignore(status);
//     return;
//   }
//
//   iree_tokenizer_regex_dfa_t dfa;
//   iree_tokenizer_regex_dfa_load(
//       iree_make_const_byte_span(dfa_data, dfa_size), &dfa);
//   // Use dfa...
//
//   iree_tokenizer_regex_compiled_free(dfa_data, iree_allocator_system());

#ifndef IREE_TOKENIZER_REGEX_COMPILE_H_
#define IREE_TOKENIZER_REGEX_COMPILE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/regex/exec.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Compilation Flags
//===----------------------------------------------------------------------===//

typedef enum iree_tokenizer_regex_compile_flag_bits_e {
  IREE_TOKENIZER_REGEX_COMPILE_FLAG_NONE = 0,
  // Enable case-insensitive matching for the entire pattern.
  // This expands all ASCII letter transitions to include both cases.
  // NOTE: Only affects ASCII letters (a-z, A-Z), not Unicode letters.
  IREE_TOKENIZER_REGEX_COMPILE_FLAG_CASE_INSENSITIVE = 1 << 0,
} iree_tokenizer_regex_compile_flag_bits_t;
typedef uint32_t iree_tokenizer_regex_compile_flags_t;

//===----------------------------------------------------------------------===//
// Compilation Errors
//===----------------------------------------------------------------------===//

// Detailed error information from compilation.
// Provides position information for error reporting/highlighting.
typedef struct iree_tokenizer_regex_compile_error_t {
  iree_host_size_t position;  // Byte offset in pattern where error occurred.
  iree_host_size_t length;    // Span of problematic text (0 if unknown).
  const char* message;        // Static error message (not allocated).
} iree_tokenizer_regex_compile_error_t;

//===----------------------------------------------------------------------===//
// Compilation API
//===----------------------------------------------------------------------===//

// Compiles a regex pattern to binary DFA format.
//
// |pattern| is the regex pattern to compile.
// |flags| controls compilation behavior.
// |allocator| is used to allocate the output buffer.
// |out_dfa_data| receives the binary DFA data.
// |out_dfa_size| receives the size of the binary data.
// |out_error| receives error details on failure (optional, may be NULL).
//
// On success, the caller owns |out_dfa_data| and must free it with
// iree_tokenizer_regex_compiled_free().
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - IREE_STATUS_INVALID_ARGUMENT for syntax errors (see out_error).
//   - IREE_STATUS_FAILED_PRECONDITION for unsupported features.
//   - IREE_STATUS_RESOURCE_EXHAUSTED if DFA state limit exceeded.
iree_status_t iree_tokenizer_regex_compile(
    iree_string_view_t pattern, iree_tokenizer_regex_compile_flags_t flags,
    iree_allocator_t allocator, uint8_t** out_dfa_data,
    iree_host_size_t* out_dfa_size,
    iree_tokenizer_regex_compile_error_t* out_error);

// Compiles a regex pattern and loads it in one call.
//
// This is a convenience function that compiles the pattern and immediately
// loads the resulting DFA. Useful when you want to compile and use a pattern
// without manually managing the intermediate binary data.
//
// |pattern| is the regex pattern to compile.
// |flags| controls compilation behavior.
// |allocator| is used to allocate the DFA storage.
// |out_dfa| receives the loaded DFA handle.
// |out_dfa_storage| receives the underlying binary data (for cleanup).
// |out_error| receives error details on failure (optional, may be NULL).
//
// On success, the caller owns |out_dfa_storage| and must free it with
// iree_tokenizer_regex_compiled_free(). The |out_dfa| points into this storage.
//
// Returns: Same as iree_tokenizer_regex_compile().
iree_status_t iree_tokenizer_regex_compile_and_load(
    iree_string_view_t pattern, iree_tokenizer_regex_compile_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_regex_dfa_t* out_dfa,
    uint8_t** out_dfa_storage, iree_tokenizer_regex_compile_error_t* out_error);

// Frees compiled DFA data.
//
// |dfa_data| is the data returned by iree_tokenizer_regex_compile().
// |allocator| must be the same allocator used during compilation.
void iree_tokenizer_regex_compiled_free(uint8_t* dfa_data,
                                        iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_REGEX_COMPILE_H_
