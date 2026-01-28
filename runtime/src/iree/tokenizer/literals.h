// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Literal token handling for tokenizers.
//
// Literal tokens are strings that map directly to token IDs, bypassing the
// normal tokenization algorithm (BPE/WordPiece). They are defined in the
// "added_tokens" array of HuggingFace tokenizer.json files.
//
// Literals support flags that control matching behavior:
// - lstrip: Consume leading whitespace before the match
// - rstrip: Consume trailing whitespace after the match
// - single_word: Only match at word boundaries
// - normalized: Match against lowercase/normalized text
//
// During encoding, literals are intercepted BEFORE the pre-tokenizer runs.
// This prevents special tokens like <mask> from being incorrectly split by
// pre-tokenizers that treat < and > as punctuation.

#ifndef IREE_TOKENIZER_LITERALS_H_
#define IREE_TOKENIZER_LITERALS_H_

#include "iree/base/api.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Literal Token Flags
//===----------------------------------------------------------------------===//

// Flags controlling literal token matching behavior.
typedef enum iree_tokenizer_literal_flag_bits_e {
  IREE_TOKENIZER_LITERAL_FLAG_NONE = 0,
  // Consume leading whitespace before the match.
  IREE_TOKENIZER_LITERAL_FLAG_LSTRIP = 1 << 0,
  // Consume trailing whitespace after the match.
  IREE_TOKENIZER_LITERAL_FLAG_RSTRIP = 1 << 1,
  // Only match at word boundaries.
  IREE_TOKENIZER_LITERAL_FLAG_SINGLE_WORD = 1 << 2,
  // Match against lowercase/normalized text.
  IREE_TOKENIZER_LITERAL_FLAG_NORMALIZED = 1 << 3,
  // Token is a special token (BOS/EOS/CLS/SEP/etc).
  IREE_TOKENIZER_LITERAL_FLAG_SPECIAL = 1 << 4,
} iree_tokenizer_literal_flag_bits_t;
typedef uint16_t iree_tokenizer_literal_flags_t;

// Flags that require interception (token must be matched before pre-tokenizer).
#define IREE_TOKENIZER_LITERAL_FLAGS_NEEDS_INTERCEPTION                      \
  (IREE_TOKENIZER_LITERAL_FLAG_LSTRIP | IREE_TOKENIZER_LITERAL_FLAG_RSTRIP | \
   IREE_TOKENIZER_LITERAL_FLAG_SINGLE_WORD |                                 \
   IREE_TOKENIZER_LITERAL_FLAG_NORMALIZED)

//===----------------------------------------------------------------------===//
// Literal Token Entry
//===----------------------------------------------------------------------===//

// A single literal token entry.
typedef struct iree_tokenizer_literal_t {
  int32_t id;
  iree_string_view_t content;  // Points into string storage.
  iree_tokenizer_literal_flags_t flags;
  iree_tokenizer_special_token_t special_type;  // -1 if not a special token.
} iree_tokenizer_literal_t;

//===----------------------------------------------------------------------===//
// Literals Collection
//===----------------------------------------------------------------------===//

// Collection of literal tokens with matching support.
typedef struct iree_tokenizer_literals_t {
  // Literal token entries.
  iree_tokenizer_literal_t* entries;
  iree_host_size_t count;
  iree_host_size_t capacity;

  // String storage (owns the content strings).
  char* string_storage;
  iree_host_size_t string_storage_size;
  iree_host_size_t string_storage_capacity;

  // Match order: indices into entries[], sorted by content length descending.
  // Used for greedy longest-match-first matching.
  iree_host_size_t* match_order;

  // True if any literals have flags requiring interception.
  bool needs_interception;

  iree_allocator_t allocator;
} iree_tokenizer_literals_t;

// Initializes an empty literals collection.
void iree_tokenizer_literals_initialize(iree_allocator_t allocator,
                                        iree_tokenizer_literals_t* literals);

// Frees all memory owned by the literals collection.
void iree_tokenizer_literals_deinitialize(iree_tokenizer_literals_t* literals);

// Adds a literal token to the collection.
// |content| is copied into internal storage.
// Skips duplicates by ID (first entry wins).
iree_status_t iree_tokenizer_literals_add(
    iree_tokenizer_literals_t* literals, int32_t id, iree_string_view_t content,
    iree_tokenizer_literal_flags_t flags,
    iree_tokenizer_special_token_t special_type);

// Finalizes the literals collection for matching.
// Must be called after all literals are added and before intercept is called.
// Sorts entries by content length for greedy matching.
iree_status_t iree_tokenizer_literals_finalize(
    iree_tokenizer_literals_t* literals);

//===----------------------------------------------------------------------===//
// Literal Matching
//===----------------------------------------------------------------------===//

// Intercepts literal tokens in text.
//
// Scans |text| for literal token matches. For each match:
// - Calls |text_callback| with any preceding non-matching text
// - Calls |token_callback| with the matched token ID
//
// After all matches, calls |text_callback| with any remaining text.
//
// Uses greedy longest-match-first strategy. Literals with lstrip/rstrip flags
// consume adjacent whitespace as part of the match.
//
// If |literals->needs_interception| is false, simply calls |text_callback|
// with the entire text (zero-overhead bypass).
iree_status_t iree_tokenizer_literals_intercept(
    const iree_tokenizer_literals_t* literals, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t text_callback,
    iree_tokenizer_token_callback_fn_t token_callback, void* user_data);

//===----------------------------------------------------------------------===//
// Literal Lookup
//===----------------------------------------------------------------------===//

// Finds a literal by ID. Returns NULL if not found.
const iree_tokenizer_literal_t* iree_tokenizer_literals_find_by_id(
    const iree_tokenizer_literals_t* literals, int32_t id);

// Returns true if the given ID is a special token in this collection.
bool iree_tokenizer_literals_is_special(
    const iree_tokenizer_literals_t* literals, int32_t id);

//===----------------------------------------------------------------------===//
// Special Token Matching
//===----------------------------------------------------------------------===//

// Matches token content against known special token patterns.
// Returns the special token type, or -1 (cast to
// iree_tokenizer_special_token_t) if not recognized.
//
// Recognizes patterns from both BPE (GPT-2/3, Llama) and BERT tokenizers:
// - UNK: [UNK], <unk>, <|unk|>
// - BOS: <s>, <bos>, [BOS], <|begin_of_text|>, <|startoftext|>
// - EOS: </s>, <eos>, [EOS], <|end_of_text|>, <|endoftext|>
// - PAD: [PAD], <pad>, <|pad|>
// - MASK: [MASK], <mask>, <|mask|>
// - CLS: [CLS], <cls>
// - SEP: [SEP], <sep>
iree_tokenizer_special_token_t iree_tokenizer_match_special_token(
    iree_string_view_t content);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_LITERALS_H_
