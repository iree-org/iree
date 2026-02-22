// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Word cache and suffix handling for BPE tokenization.
//
// The word cache provides direct-mapped hash table memoization for
// previously-tokenized segments. This exploits Zipf's law: common words
// (the, a, is, etc.) appear frequently and can bypass the BPE merge
// computation entirely on cache hit.
//
// Suffix handling supports models like CLIP that use end-of-word suffixes
// (e.g., "</w>") to distinguish word-final vs word-internal tokens.

#include <string.h>

#include "iree/tokenizer/byte_level_tables.h"
#include "iree/tokenizer/model/bpe_internal.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

//===----------------------------------------------------------------------===//
// Word Cache Helpers
//===----------------------------------------------------------------------===//

// Attempts to serve a segment from the word cache. Returns true if the cache
// contained this segment AND all tokens were successfully emitted. Returns
// false on cache miss, oversized segments, or insufficient output capacity.
//
// When byte offsets are requested (cursor->offset_ptr != NULL), the cache is
// bypassed because it does not store per-token byte positions within segments.
bool iree_tokenizer_bpe_cache_lookup(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment, iree_tokenizer_bpe_output_cursor_t* cursor) {
  if (model->cache_capacity == 0) return false;
  if (cursor->offset_ptr) return false;
  if (segment.size > IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES) return false;

  uint32_t hash =
      iree_tokenizer_bpe_cache_hash((const uint8_t*)segment.data, segment.size);
  iree_tokenizer_bpe_cache_entry_t* cache =
      iree_tokenizer_bpe_state_cache(state, model);
  iree_tokenizer_bpe_cache_entry_t* entry =
      &cache[hash & model->cache_capacity_mask];

  if (entry->key_hash != hash || entry->key_length != (uint16_t)segment.size ||
      memcmp(entry->key, segment.data, segment.size) != 0) {
    return false;
  }

  // Verify capacity before emitting to avoid partial emission.
  if (cursor->remaining < entry->token_count) return false;
  for (uint16_t t = 0; t < entry->token_count; ++t) {
    iree_tokenizer_bpe_emit_and_track(state, cursor, entry->tokens[t], 0,
                                      (uint32_t)segment.size);
  }
  return true;
}

// Populates a word cache entry from the backtrack stack after encoding.
// Silently skipped when the segment or token count exceeds cache limits.
void iree_tokenizer_bpe_cache_populate(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment,
    const iree_tokenizer_bpe_backtrack_entry_t* stack,
    iree_host_size_t token_count) {
  if (model->cache_capacity == 0) return;
  if (segment.size > IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES) return;
  if (token_count > IREE_TOKENIZER_BPE_CACHE_MAX_TOKENS) return;

  uint32_t hash =
      iree_tokenizer_bpe_cache_hash((const uint8_t*)segment.data, segment.size);
  iree_tokenizer_bpe_cache_entry_t* cache =
      iree_tokenizer_bpe_state_cache(state, model);
  iree_tokenizer_bpe_cache_entry_t* entry =
      &cache[hash & model->cache_capacity_mask];

  entry->key_hash = hash;
  entry->key_length = (uint16_t)segment.size;
  entry->token_count = (uint16_t)token_count;
  memcpy(entry->key, segment.data, segment.size);
  for (iree_host_size_t i = 0; i < token_count; ++i) {
    entry->tokens[i] = stack[i].token_id;
  }
}

//===----------------------------------------------------------------------===//
// Trie Helpers with ByteLevel Support
//===----------------------------------------------------------------------===//

// Helper: walks segment bytes through trie cursor with ByteLevel support.
// Returns true if all bytes were successfully advanced, false otherwise.
static bool iree_tokenizer_bpe_trie_walk_segment_bytes(
    const iree_tokenizer_bpe_model_t* model,
    iree_tokenizer_trie_cursor_t* cursor, const uint8_t* data,
    iree_host_size_t length) {
  const bool byte_level =
      iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);

  for (iree_host_size_t i = 0; i < length; ++i) {
    uint8_t byte = data[i];
    if (!byte_level || (byte >= 0x21 && byte <= 0x7E)) {
      // Non-ByteLevel or printable ASCII: identity mapping (single UTF-8 byte).
      if (!iree_tokenizer_trie_cursor_advance(cursor, byte)) {
        return false;
      }
    } else {
      // ByteLevel with non-printable or high byte: use pre-computed UTF-8.
      const iree_tokenizer_byte_level_utf8_t* utf8 =
          &iree_tokenizer_byte_level_utf8[byte];
      if (!iree_tokenizer_trie_cursor_advance(cursor, utf8->bytes[0])) {
        return false;
      }
      if (utf8->length == 2 &&
          !iree_tokenizer_trie_cursor_advance(cursor, utf8->bytes[1])) {
        return false;
      }
    }
  }
  return true;
}

// Checks if segment (with ByteLevel transformation) + suffix (raw) matches a
// token. The segment bytes are transformed via the ByteLevel lookup table when
// applicable, while suffix bytes are walked directly since suffixes like "</w>"
// are already raw UTF-8 in the vocabulary.
//
// This is used for models with end_of_word_suffix to efficiently check if the
// entire segment plus suffix matches a single vocabulary token, without
// allocating a buffer to concatenate them.
//
// Returns the token ID and total length (segment.size + suffix.size), or -1
// and 0 if no exact match at the final position.
void iree_tokenizer_bpe_trie_longest_match_with_suffix(
    const iree_tokenizer_bpe_model_t* model, iree_string_view_t segment,
    iree_string_view_t suffix, int32_t* out_token_id,
    iree_host_size_t* out_length) {
  *out_token_id = -1;
  *out_length = 0;

  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  // Phase 1: Walk segment bytes with ByteLevel transformation.
  if (!iree_tokenizer_bpe_trie_walk_segment_bytes(
          model, &cursor, (const uint8_t*)segment.data, segment.size)) {
    return;  // No match possible.
  }

  // Phase 2: Walk suffix bytes directly (no transformation - suffix is raw
  // UTF-8).
  for (iree_host_size_t i = 0; i < suffix.size; ++i) {
    if (!iree_tokenizer_trie_cursor_advance(&cursor, (uint8_t)suffix.data[i])) {
      return;  // No match with suffix.
    }
  }

  // Check for token at segment+suffix boundary.
  int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
  if (token_id >= 0) {
    *out_token_id = token_id;
    *out_length = segment.size + suffix.size;
  }
}

// Finds the longest token in the trie with ByteLevel transformation applied.
// Unlike iree_tokenizer_bpe_trie_longest_match() which walks raw bytes, this
// function transforms segment bytes via the ByteLevel lookup table when the
// BYTE_LEVEL_INPUT flag is set.
//
// Used for IGNORE_MERGES mode where we need longest-match vocab lookup but
// still need to respect ByteLevel byte-to-character mapping.
void iree_tokenizer_bpe_trie_longest_match_byte_level(
    const iree_tokenizer_bpe_model_t* model, iree_string_view_t segment,
    int32_t* out_token_id, iree_host_size_t* out_length) {
  *out_token_id = -1;
  *out_length = 0;

  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  const bool byte_level =
      iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);

  for (iree_host_size_t i = 0; i < segment.size; ++i) {
    uint8_t byte = (uint8_t)segment.data[i];
    bool advanced = false;

    if (!byte_level || (byte >= 0x21 && byte <= 0x7E)) {
      // Non-ByteLevel or printable ASCII: identity mapping.
      advanced = iree_tokenizer_trie_cursor_advance(&cursor, byte);
    } else {
      // ByteLevel with non-printable or high byte: use pre-computed UTF-8.
      const iree_tokenizer_byte_level_utf8_t* utf8 =
          &iree_tokenizer_byte_level_utf8[byte];
      advanced = iree_tokenizer_trie_cursor_advance(&cursor, utf8->bytes[0]);
      if (advanced && utf8->length == 2) {
        advanced = iree_tokenizer_trie_cursor_advance(&cursor, utf8->bytes[1]);
      }
    }

    if (!advanced) break;  // No more matches possible.

    int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (token_id >= 0) {
      // Found a token ending here - remember it (greedy longest match).
      *out_token_id = token_id;
      *out_length = i + 1;
    }
  }
}

//===----------------------------------------------------------------------===//
// End-of-Word Suffix Handling
//===----------------------------------------------------------------------===//

// Forward declaration for mutual tail calls.
static void iree_tokenizer_bpe_apply_suffix_to_last_token(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment);

// Checks if a token has a suffixed version in the vocabulary.
// Returns true if token_text + suffix exists as a token, false otherwise.
// Used to decide whether splitting a merge would help with suffix application.
static bool iree_tokenizer_bpe_token_has_suffix(
    const iree_tokenizer_bpe_model_t* model, uint32_t token_id) {
  if (model->end_of_word_suffix_length == 0) return false;

  // Get the token's text from the vocabulary.
  iree_string_view_t token_text =
      iree_tokenizer_vocab_token_text(model->vocab, token_id);
  if (token_text.size == 0) return false;

  // Walk the token text through the trie.
  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  // For ByteLevel mode, the token text is already in ByteLevel form (UTF-8),
  // so we can walk it directly without conversion.
  for (iree_host_size_t i = 0; i < token_text.size; ++i) {
    if (!iree_tokenizer_trie_cursor_advance(&cursor,
                                            (uint8_t)token_text.data[i])) {
      return false;
    }
  }

  // Walk the suffix through the trie.
  for (iree_host_size_t i = 0; i < model->end_of_word_suffix_length; ++i) {
    if (!iree_tokenizer_trie_cursor_advance(
            &cursor, (uint8_t)model->end_of_word_suffix[i])) {
      return false;
    }
  }

  return iree_tokenizer_trie_cursor_token_id(&cursor) >= 0;
}

// Handles the case where the last token has no suffixed version but is a merge
// result. For example, with CLIP, token 'Ã¥' (23176) exists but 'Ã¥</w>' does
// not. In this case, the merge should NOT have happened at end-of-word. Split
// the token back into its components and retry suffix application on the right
// component. This produces 'Ã' + '¥</w>' (tokens 127, 354) instead of 'Ã¥'.
//
// Only splits if the right component either:
// - Has a suffixed version (splitting will succeed), or
// - Is itself a merge result (further splitting might find a suffixable
// component). Otherwise, leaves the merged token as-is (neither it nor its
// parts can be suffixed, so keeping the merge is preferable).
static void iree_tokenizer_bpe_try_split_for_suffix(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment) {
  iree_tokenizer_bpe_backtrack_entry_t* stack =
      iree_tokenizer_bpe_state_backtrack_stack(state, model);
  iree_tokenizer_bpe_backtrack_entry_t* last_entry =
      &stack[state->backtrack.stack_count - 1];

  const iree_tokenizer_bpe_split_entry_t* split =
      &model->backtrack_tables.split_table[last_entry->token_id];

  // Check if token is a merge result (not a base token).
  if (split->left_id == (uint32_t)last_entry->token_id) {
    return;  // Base token, nothing to split.
  }

  // Check if splitting would help: does right component have a suffix OR
  // is right component itself a merge that could be further split?
  const iree_tokenizer_bpe_split_entry_t* right_split =
      &model->backtrack_tables.split_table[split->right_id];
  bool right_is_merge = (right_split->left_id != split->right_id);
  bool right_has_suffix =
      iree_tokenizer_bpe_token_has_suffix(model, split->right_id);
  if (!right_has_suffix && !right_is_merge) {
    // Neither the right component nor its descendants can be suffixed.
    // Leave the merged token as-is.
    return;
  }

  // Token is a merge result and splitting might help. Do it.
  // Walk through segment bytes to find where the left component ends.
  uint32_t last_entry_start =
      iree_tokenizer_bpe_backtrack_entry_start_byte(last_entry);
  const uint8_t* last_token_start =
      (const uint8_t*)segment.data + last_entry_start;
  iree_host_size_t last_token_length = segment.size - last_entry_start;

  iree_host_size_t left_raw_bytes = 0;
  {
    bool byte_level =
        (model->flags & IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT) != 0;
    iree_tokenizer_trie_cursor_t left_cursor;
    iree_tokenizer_trie_cursor_reset(&left_cursor, model->trie);
    for (iree_host_size_t i = 0; i < last_token_length; ++i) {
      if (!iree_tokenizer_bpe_trie_advance_byte(
              &left_cursor, last_token_start[i], byte_level)) {
        break;
      }
      int32_t tid = iree_tokenizer_trie_cursor_token_id(&left_cursor);
      if (tid == (int32_t)split->left_id) {
        left_raw_bytes = i + 1;
        break;
      }
    }
  }

  if (left_raw_bytes > 0 && left_raw_bytes < last_token_length) {
    // Update stack: replace merged token with left, push right.
    uint32_t last_start_byte =
        iree_tokenizer_bpe_backtrack_entry_start_byte(last_entry);
    last_entry->token_id = split->left_id;
    // Push right component.
    IREE_ASSERT(state->backtrack.stack_count < model->backtrack_stack_capacity);
    iree_tokenizer_bpe_backtrack_entry_set(
        &stack[state->backtrack.stack_count], split->right_id,
        last_start_byte + (uint32_t)left_raw_bytes, 0);
    state->backtrack.stack_count++;
    // Apply suffix to the new last token (right component).
    iree_tokenizer_bpe_apply_suffix_to_last_token(model, state, segment);
    return;
  }
}

// For models with end_of_word_suffix (e.g., CLIP's "</w>"), applies suffix
// handling to the end of the token sequence.
//
// This handles three cases with increasing complexity:
// 1. Simple suffix: replace last token with its suffixed version
//    Example: [bb, bb] -> [bb, bb</w>] when "bb</w>" exists
// 2. Suffix merge: merge last two tokens when the result with suffix exists
//    Example: [caf, Ã©] -> [cafÃ©</w>] when merge(caf, Ã©</w>) exists
// 3. Recursive merge: repeatedly merge previous token with current suffixed
//    token until no more merges are possible
//    Example: [a, b, c] + suffix -> [a, bc</w>] -> [abc</w>] when both merges
//    exist
//
// The merge case handles CLIP-style vocabularies where merges like
// ['caf', 'Ã©</w>'] produce 'cafÃ©</w>' but no merge exists for ['caf', 'Ã©'].
//
// Recursive merging is critical for cases like CLIP's 10-emoji input where:
// - Backtracking produces [..., 4310, 4310, 4310] (3 single emojis at end)
// - suffix(4310) = 3986 (🎉</w>)
// - merge(4310, 3986) = 13450 (🎉🎉</w>)
// - merge(4310, 13450) = 20828 (🎉🎉🎉</w>)
// Without recursive merging, we'd stop at 13450 and miss 20828.
//
// For ByteLevel mode, raw segment bytes are converted to their ByteLevel UTF-8
// form before trie traversal.
static void iree_tokenizer_bpe_apply_suffix_to_last_token(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment) {
  if (state->backtrack.stack_count == 0) return;

  iree_tokenizer_bpe_backtrack_entry_t* stack =
      iree_tokenizer_bpe_state_backtrack_stack(state, model);
  iree_tokenizer_bpe_backtrack_entry_t* last_entry =
      &stack[state->backtrack.stack_count - 1];

  // First, find the suffixed version of the last token.
  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  uint32_t last_entry_start =
      iree_tokenizer_bpe_backtrack_entry_start_byte(last_entry);
  const uint8_t* last_token_start =
      (const uint8_t*)segment.data + last_entry_start;
  iree_host_size_t last_token_length = segment.size - last_entry_start;

  if (!iree_tokenizer_bpe_trie_walk_segment_bytes(
          model, &cursor, last_token_start, last_token_length)) {
    return;  // Last token bytes don't match trie (shouldn't happen).
  }

  // Walk suffix bytes. Suffix is raw (not ByteLevel transformed).
  bool suffix_walk_ok = true;
  for (iree_host_size_t i = 0; i < model->end_of_word_suffix_length; ++i) {
    if (!iree_tokenizer_trie_cursor_advance(
            &cursor, (uint8_t)model->end_of_word_suffix[i])) {
      suffix_walk_ok = false;
      break;
    }
  }

  int32_t suffixed_last_token_id =
      suffix_walk_ok ? iree_tokenizer_trie_cursor_token_id(&cursor) : -1;

  // No suffixed version exists for this token. Try splitting if it's a merge.
  if (suffixed_last_token_id < 0) {
    iree_tokenizer_bpe_try_split_for_suffix(model, state, segment);
    return;
  }

  // Case 1: Simple suffix replacement (when no previous token or no merge).
  last_entry->token_id = suffixed_last_token_id;

  // Case 2 & 3: Try merging with previous tokens.
  // Repeatedly merge (previous, current_suffixed) until no merge exists.
  // This handles both simple case (one merge) and recursive case (N merges).
  while (state->backtrack.stack_count >= 2) {
    iree_tokenizer_bpe_backtrack_entry_t* prev_entry =
        &stack[state->backtrack.stack_count - 2];
    iree_tokenizer_bpe_backtrack_entry_t* curr_entry =
        &stack[state->backtrack.stack_count - 1];

    iree_tokenizer_merge_hash_result_t merge =
        iree_tokenizer_vocab_merge_hash_lookup(
            model->merge_hash, prev_entry->token_id, curr_entry->token_id);
    if (!iree_tokenizer_merge_hash_result_is_valid(merge)) {
      break;  // No merge possible, done.
    }

    // Merge found! Replace previous two tokens with merged result.
    prev_entry->token_id = merge.result_id;
    state->backtrack.stack_count--;
  }
}

// External-facing wrapper to call the static function.
// This allows the encode state machine to apply suffix handling.
void iree_tokenizer_bpe_apply_suffix_to_backtrack(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment) {
  iree_tokenizer_bpe_apply_suffix_to_last_token(model, state, segment);
}

// Same as apply_suffix_to_last_token but operates on the sliding window
// instead of the backtrack stack. Called during FLUSH phase when the
// BYTE_LOOP path was used for long segments (> max_backtrack_segment_bytes).
//
// This handles models with end_of_word_suffix (like CLIP's "</w>") that need
// the final token replaced with its suffixed version. The algorithm:
//   1. Find the suffixed version of the last window token via trie lookup.
//   2. Repeatedly merge with previous window tokens while merges exist.
//
// Example with segment "test" and suffix "</w>":
//   Window before: [t, e, s, t]
//   After suffix lookup: [t, e, s, t</w>]
//   After merges: [test</w>] (if merge chain exists)
void iree_tokenizer_bpe_apply_suffix_to_last_window_token(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment) {
  if (state->window.count == 0) return;

  iree_tokenizer_bpe_window_token_t* last_token =
      iree_tokenizer_bpe_window_at(state, model, state->window.count - 1);

  // Find the suffixed version of the last token via trie lookup.
  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  const uint8_t* last_token_start =
      (const uint8_t*)segment.data + last_token->start_byte;
  iree_host_size_t last_token_length =
      last_token->end_byte - last_token->start_byte;

  if (!iree_tokenizer_bpe_trie_walk_segment_bytes(
          model, &cursor, last_token_start, last_token_length)) {
    return;  // Last token bytes don't match trie (shouldn't happen).
  }

  // Walk suffix bytes. Suffix is raw (not ByteLevel transformed).
  for (iree_host_size_t i = 0; i < model->end_of_word_suffix_length; ++i) {
    if (!iree_tokenizer_trie_cursor_advance(
            &cursor, (uint8_t)model->end_of_word_suffix[i])) {
      return;  // No suffixed token exists.
    }
  }

  int32_t suffixed_last_token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
  if (suffixed_last_token_id < 0) {
    return;  // No suffixed version of last token.
  }

  // Replace the last token with its suffixed version.
  last_token->token_id = suffixed_last_token_id;

  // Repeatedly merge with previous tokens while merges exist.
  // This handles cases where suffix enables new merge chains.
  while (state->window.count >= 2) {
    iree_tokenizer_bpe_window_token_t* prev_token =
        iree_tokenizer_bpe_window_at(state, model, state->window.count - 2);
    iree_tokenizer_bpe_window_token_t* curr_token =
        iree_tokenizer_bpe_window_at(state, model, state->window.count - 1);

    iree_tokenizer_merge_hash_result_t merge =
        iree_tokenizer_vocab_merge_hash_lookup(
            model->merge_hash, prev_token->token_id, curr_token->token_id);
    if (!iree_tokenizer_merge_hash_result_is_valid(merge)) {
      break;  // No merge possible, done.
    }

    // Merge found! Combine tokens: update prev with merged result, remove curr.
    prev_token->token_id = merge.result_id;
    prev_token->end_byte = curr_token->end_byte;
    state->window.count--;
  }
}
