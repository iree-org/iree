// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/literals.h"

//===----------------------------------------------------------------------===//
// Literals Collection
//===----------------------------------------------------------------------===//

void iree_tokenizer_literals_initialize(iree_allocator_t allocator,
                                        iree_tokenizer_literals_t* literals) {
  memset(literals, 0, sizeof(*literals));
  literals->allocator = allocator;
}

void iree_tokenizer_literals_deinitialize(iree_tokenizer_literals_t* literals) {
  if (literals->entries) {
    iree_allocator_free(literals->allocator, literals->entries);
  }
  if (literals->string_storage) {
    iree_allocator_free(literals->allocator, literals->string_storage);
  }
  if (literals->match_order) {
    iree_allocator_free(literals->allocator, literals->match_order);
  }
  memset(literals, 0, sizeof(*literals));
}

iree_status_t iree_tokenizer_literals_add(
    iree_tokenizer_literals_t* literals, int32_t id, iree_string_view_t content,
    iree_tokenizer_literal_flags_t flags,
    iree_tokenizer_special_token_t special_type) {
  // Skip duplicates by ID (first entry wins).
  for (iree_host_size_t i = 0; i < literals->count; ++i) {
    if (literals->entries[i].id == id) {
      return iree_ok_status();
    }
  }

  // Skip empty content.
  if (content.size == 0) {
    return iree_ok_status();
  }

  // Grow entries array if needed.
  if (literals->count >= literals->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        literals->allocator, /*min_capacity=*/16,
        sizeof(iree_tokenizer_literal_t), &literals->capacity,
        (void**)&literals->entries));
  }

  // Grow string storage if needed.
  iree_host_size_t new_string_size =
      literals->string_storage_size + content.size;
  if (new_string_size > literals->string_storage_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        literals->allocator, iree_max(256, new_string_size),
        /*element_size=*/1, &literals->string_storage_capacity,
        (void**)&literals->string_storage));

    // Update all existing content pointers (they may have moved).
    char* base = literals->string_storage;
    iree_host_size_t offset = 0;
    for (iree_host_size_t i = 0; i < literals->count; ++i) {
      literals->entries[i].content.data = base + offset;
      offset += literals->entries[i].content.size;
    }
  }

  // Copy content into string storage.
  char* content_ptr = literals->string_storage + literals->string_storage_size;
  memcpy(content_ptr, content.data, content.size);
  literals->string_storage_size += content.size;

  // Track if any literals need interception.
  if (flags & IREE_TOKENIZER_LITERAL_FLAGS_NEEDS_INTERCEPTION) {
    literals->needs_interception = true;
  }

  // Add the entry.
  literals->entries[literals->count] = (iree_tokenizer_literal_t){
      .id = id,
      .content = iree_make_string_view(content_ptr, content.size),
      .flags = flags,
      .special_type = special_type,
  };
  literals->count++;

  return iree_ok_status();
}

iree_status_t iree_tokenizer_literals_finalize(
    iree_tokenizer_literals_t* literals) {
  if (literals->count == 0) {
    return iree_ok_status();
  }

  // Allocate match order array.
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      literals->allocator, literals->count * sizeof(iree_host_size_t),
      (void**)&literals->match_order));

  // Initialize with sequential indices.
  for (iree_host_size_t i = 0; i < literals->count; ++i) {
    literals->match_order[i] = i;
  }

  // Sort by content length descending using insertion sort.
  // Literal count is typically small (< 100), so O(n^2) is acceptable.
  for (iree_host_size_t i = 1; i < literals->count; ++i) {
    iree_host_size_t key = literals->match_order[i];
    iree_host_size_t key_length = literals->entries[key].content.size;
    iree_host_size_t j = i;
    while (j > 0) {
      iree_host_size_t prev = literals->match_order[j - 1];
      iree_host_size_t prev_length = literals->entries[prev].content.size;
      if (prev_length >= key_length) break;  // prev is longer or equal, stop.
      literals->match_order[j] = prev;
      j--;
    }
    literals->match_order[j] = key;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Literal Matching Helpers
//===----------------------------------------------------------------------===//

// Returns true if the character is whitespace.
static bool iree_tokenizer_is_whitespace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Returns true if the character is a word boundary (non-alphanumeric).
static bool iree_tokenizer_is_word_boundary(char c) {
  return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_');
}

// Converts ASCII to lowercase.
static char iree_tokenizer_ascii_tolower(char c) {
  if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
  return c;
}

// Checks if literal matches at position with flag handling.
// Returns true if match found, and sets match_start/match_end to the consumed
// range (which may include whitespace for lstrip/rstrip).
static bool iree_tokenizer_literal_matches_at(
    const iree_tokenizer_literal_t* literal, iree_string_view_t text,
    iree_host_size_t position, iree_host_size_t* out_match_start,
    iree_host_size_t* out_match_end) {
  iree_host_size_t match_start = position;
  iree_host_size_t match_end = position;

  // Handle lstrip: look for whitespace before the position that we can consume.
  if (literal->flags & IREE_TOKENIZER_LITERAL_FLAG_LSTRIP) {
    // Scan backwards from position to find whitespace to consume.
    iree_host_size_t ws_start = position;
    while (ws_start > 0 &&
           iree_tokenizer_is_whitespace(text.data[ws_start - 1])) {
      ws_start--;
    }
    match_start = ws_start;
  }

  // Check if the literal content matches at position.
  if (position + literal->content.size > text.size) {
    return false;  // Not enough text remaining.
  }

  bool content_matches;
  if (literal->flags & IREE_TOKENIZER_LITERAL_FLAG_NORMALIZED) {
    // Case-insensitive comparison (ASCII only for now).
    content_matches = true;
    for (iree_host_size_t i = 0; i < literal->content.size; ++i) {
      if (iree_tokenizer_ascii_tolower(text.data[position + i]) !=
          iree_tokenizer_ascii_tolower(literal->content.data[i])) {
        content_matches = false;
        break;
      }
    }
  } else {
    // Exact match.
    content_matches = memcmp(text.data + position, literal->content.data,
                             literal->content.size) == 0;
  }

  if (!content_matches) {
    return false;
  }

  match_end = position + literal->content.size;

  // Handle rstrip: consume trailing whitespace.
  if (literal->flags & IREE_TOKENIZER_LITERAL_FLAG_RSTRIP) {
    while (match_end < text.size &&
           iree_tokenizer_is_whitespace(text.data[match_end])) {
      match_end++;
    }
  }

  // Handle single_word: check word boundaries.
  if (literal->flags & IREE_TOKENIZER_LITERAL_FLAG_SINGLE_WORD) {
    // Check character before match (must be word boundary or start of text).
    if (position > 0 &&
        !iree_tokenizer_is_word_boundary(text.data[position - 1])) {
      return false;
    }
    // Check character after match (must be word boundary or end of text).
    iree_host_size_t content_end = position + literal->content.size;
    if (content_end < text.size &&
        !iree_tokenizer_is_word_boundary(text.data[content_end])) {
      return false;
    }
  }

  *out_match_start = match_start;
  *out_match_end = match_end;
  return true;
}

//===----------------------------------------------------------------------===//
// Literal Matching
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_literals_intercept(
    const iree_tokenizer_literals_t* literals, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t text_callback,
    iree_tokenizer_token_callback_fn_t token_callback, void* user_data) {
  // Fast path: no interception needed.
  if (!literals->needs_interception || literals->count == 0) {
    if (text.size > 0) {
      iree_string_view_t views[1] = {text};
      iree_string_view_list_t list = {.count = 1, .values = views};
      return text_callback(user_data, list);
    }
    return iree_ok_status();
  }

  iree_host_size_t position = 0;
  iree_host_size_t text_start = 0;

  while (position < text.size) {
    bool matched = false;

    // Try each literal (longest first via match_order).
    for (iree_host_size_t i = 0; i < literals->count && !matched; ++i) {
      iree_host_size_t idx = literals->match_order[i];
      const iree_tokenizer_literal_t* lit = &literals->entries[idx];

      // Only intercept literals that have interception flags.
      if (!(lit->flags & IREE_TOKENIZER_LITERAL_FLAGS_NEEDS_INTERCEPTION)) {
        continue;
      }

      iree_host_size_t match_start = position;
      iree_host_size_t match_end = position;

      if (iree_tokenizer_literal_matches_at(lit, text, position, &match_start,
                                            &match_end)) {
        // Emit preceding text (from text_start to match_start).
        if (text_start < match_start) {
          iree_string_view_t segment = iree_string_view_substr(
              text, text_start, match_start - text_start);
          iree_string_view_t views[1] = {segment};
          iree_string_view_list_t list = {.count = 1, .values = views};
          IREE_RETURN_IF_ERROR(text_callback(user_data, list));
        }

        // Emit the matched token ID.
        int32_t token_id = lit->id;
        iree_tokenizer_id_list_t ids = {.count = 1, .values = &token_id};
        IREE_RETURN_IF_ERROR(token_callback(user_data, ids));

        position = match_end;
        text_start = match_end;
        matched = true;
      }
    }

    if (!matched) {
      ++position;
    }
  }

  // Emit remaining text.
  if (text_start < text.size) {
    iree_string_view_t segment =
        iree_string_view_substr(text, text_start, text.size - text_start);
    iree_string_view_t views[1] = {segment};
    iree_string_view_list_t list = {.count = 1, .values = views};
    IREE_RETURN_IF_ERROR(text_callback(user_data, list));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Literal Lookup
//===----------------------------------------------------------------------===//

const iree_tokenizer_literal_t* iree_tokenizer_literals_find_by_id(
    const iree_tokenizer_literals_t* literals, int32_t id) {
  for (iree_host_size_t i = 0; i < literals->count; ++i) {
    if (literals->entries[i].id == id) {
      return &literals->entries[i];
    }
  }
  return NULL;
}

bool iree_tokenizer_literals_is_special(
    const iree_tokenizer_literals_t* literals, int32_t id) {
  const iree_tokenizer_literal_t* lit =
      iree_tokenizer_literals_find_by_id(literals, id);
  return lit && (lit->flags & IREE_TOKENIZER_LITERAL_FLAG_SPECIAL);
}

//===----------------------------------------------------------------------===//
// Special Token Matching
//===----------------------------------------------------------------------===//

iree_tokenizer_special_token_t iree_tokenizer_match_special_token(
    iree_string_view_t content) {
  // UNK tokens (BERT, SentencePiece, GPT).
  if (iree_string_view_equal(content, IREE_SV("[UNK]")) ||
      iree_string_view_equal(content, IREE_SV("<unk>")) ||
      iree_string_view_equal(content, IREE_SV("<|unk|>"))) {
    return IREE_TOKENIZER_SPECIAL_TOKEN_UNK;
  }

  // BOS tokens (SentencePiece, Llama, GPT).
  if (iree_string_view_equal(content, IREE_SV("<s>")) ||
      iree_string_view_equal(content, IREE_SV("<bos>")) ||
      iree_string_view_equal(content, IREE_SV("[BOS]")) ||
      iree_string_view_equal(content, IREE_SV("<|begin_of_text|>")) ||
      iree_string_view_equal(content, IREE_SV("<|startoftext|>"))) {
    return IREE_TOKENIZER_SPECIAL_TOKEN_BOS;
  }

  // EOS tokens (SentencePiece, Llama, GPT).
  if (iree_string_view_equal(content, IREE_SV("</s>")) ||
      iree_string_view_equal(content, IREE_SV("<eos>")) ||
      iree_string_view_equal(content, IREE_SV("[EOS]")) ||
      iree_string_view_equal(content, IREE_SV("<|end_of_text|>")) ||
      iree_string_view_equal(content, IREE_SV("<|endoftext|>"))) {
    return IREE_TOKENIZER_SPECIAL_TOKEN_EOS;
  }

  // PAD tokens (BERT, GPT).
  if (iree_string_view_equal(content, IREE_SV("[PAD]")) ||
      iree_string_view_equal(content, IREE_SV("<pad>")) ||
      iree_string_view_equal(content, IREE_SV("<|pad|>"))) {
    return IREE_TOKENIZER_SPECIAL_TOKEN_PAD;
  }

  // MASK tokens (BERT, GPT).
  if (iree_string_view_equal(content, IREE_SV("[MASK]")) ||
      iree_string_view_equal(content, IREE_SV("<mask>")) ||
      iree_string_view_equal(content, IREE_SV("<|mask|>"))) {
    return IREE_TOKENIZER_SPECIAL_TOKEN_MASK;
  }

  // CLS tokens (BERT, Funnel).
  if (iree_string_view_equal(content, IREE_SV("[CLS]")) ||
      iree_string_view_equal(content, IREE_SV("<cls>"))) {
    return IREE_TOKENIZER_SPECIAL_TOKEN_CLS;
  }

  // SEP tokens (BERT, Funnel).
  if (iree_string_view_equal(content, IREE_SV("[SEP]")) ||
      iree_string_view_equal(content, IREE_SV("<sep>"))) {
    return IREE_TOKENIZER_SPECIAL_TOKEN_SEP;
  }

  return (iree_tokenizer_special_token_t)-1;
}
