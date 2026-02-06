// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Parser for HuggingFace tokenizer.json added_tokens array.
//
// Added tokens are special tokens that extend the base vocabulary. Each token
// has attributes controlling matching and normalization behavior:
//
//   {
//     "id": 0,
//     "content": "[PAD]",
//     "single_word": false,
//     "lstrip": false,
//     "rstrip": false,
//     "normalized": false,
//     "special": true
//   }
//
// Fields:
//   - id: Required. The token ID in the vocabulary.
//   - content: Required. The token string content.
//   - single_word: Match only whole words, not substrings. Default: false.
//   - lstrip: Strip whitespace on the left when matching. Default: false.
//   - rstrip: Strip whitespace on the right when matching. Default: false.
//   - normalized: Apply normalizer before matching. Default: true (!special).
//   - special: Mark as special token (skipped during decode). Default: false.

#ifndef IREE_TOKENIZER_FORMAT_HUGGINGFACE_ADDED_TOKENS_JSON_H_
#define IREE_TOKENIZER_FORMAT_HUGGINGFACE_ADDED_TOKENS_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Added Token Structure
//===----------------------------------------------------------------------===//

// Flags for added token behavior.
enum iree_tokenizer_huggingface_added_token_flag_bits_e {
  IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_NONE = 0u,
  // Match only whole words, not substrings.
  IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SINGLE_WORD = 1u << 0,
  // Strip whitespace on the left when matching.
  IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_LSTRIP = 1u << 1,
  // Strip whitespace on the right when matching.
  IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_RSTRIP = 1u << 2,
  // Apply normalizer before matching.
  IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_NORMALIZED = 1u << 3,
  // Mark as special token (skipped during decode).
  IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL = 1u << 4,
};
typedef uint32_t iree_tokenizer_huggingface_added_token_flags_t;

// A single added token with its metadata.
// Content string is stored separately in a string pool.
typedef struct iree_tokenizer_huggingface_added_token_t {
  // Token ID in the vocabulary.
  iree_tokenizer_token_id_t id;
  // Behavior flags.
  iree_tokenizer_huggingface_added_token_flags_t flags;
  // Offset into the string pool where content starts.
  uint32_t content_offset;
  // Length of the content string (not including NUL terminator).
  uint32_t content_length;
} iree_tokenizer_huggingface_added_token_t;

//===----------------------------------------------------------------------===//
// Added Tokens List
//===----------------------------------------------------------------------===//

// A list of added tokens parsed from tokenizer.json.
//
// Memory layout (single allocation, naturally aligned):
//   [iree_tokenizer_huggingface_added_token_t tokens[count]][char
//   string_pool[]]
//
// The string pool contains NUL-terminated content strings. Each token's
// content_offset points into this pool.
typedef struct iree_tokenizer_huggingface_added_tokens_t {
  // Number of tokens in the list.
  iree_host_size_t count;
  // Total size of the string pool in bytes.
  iree_host_size_t string_pool_size;
  // Allocator used for the allocation (stored for cleanup).
  iree_allocator_t allocator;
  // Array of |count| token structs (fixed-size, naturally aligned).
  const iree_tokenizer_huggingface_added_token_t* tokens;
  // String pool containing all content strings.
  const char* string_pool;
} iree_tokenizer_huggingface_added_tokens_t;

//===----------------------------------------------------------------------===//
// Lifetime
//===----------------------------------------------------------------------===//

// Frees the added tokens list and all associated memory.
void iree_tokenizer_huggingface_added_tokens_free(
    iree_tokenizer_huggingface_added_tokens_t* tokens);

//===----------------------------------------------------------------------===//
// Accessors
//===----------------------------------------------------------------------===//

// Returns the token at the given index.
// Index must be < count.
static inline const iree_tokenizer_huggingface_added_token_t*
iree_tokenizer_huggingface_added_tokens_get(
    const iree_tokenizer_huggingface_added_tokens_t* list,
    iree_host_size_t index) {
  return &list->tokens[index];
}

// Returns a view of the token content string.
static inline iree_string_view_t iree_tokenizer_huggingface_added_token_content(
    const iree_tokenizer_huggingface_added_tokens_t* list,
    const iree_tokenizer_huggingface_added_token_t* token) {
  return iree_make_string_view(list->string_pool + token->content_offset,
                               token->content_length);
}

//===----------------------------------------------------------------------===//
// Added Tokens Parser
//===----------------------------------------------------------------------===//

// Parses the added_tokens array from a tokenizer.json root object.
//
// |json_root| is the full tokenizer.json object (must include braces).
// |allocator| is used for all allocations; stored for cleanup.
// |out_tokens| receives the parsed tokens on success.
//
// If the added_tokens field is absent or null, returns success with count=0.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON or invalid field values
iree_status_t iree_tokenizer_huggingface_parse_added_tokens_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_huggingface_added_tokens_t* out_tokens);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_HUGGINGFACE_ADDED_TOKENS_JSON_H_
