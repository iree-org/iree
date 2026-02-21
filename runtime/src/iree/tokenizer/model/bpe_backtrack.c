// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE backtracking algorithm for O(n) tokenization of segments.
//
// This file contains the backtracking path for BPE encoding:
// - Pair validation to check if two tokens can legally appear adjacent
// - Suffix blocking detection to prevent suboptimal tokenizations
// - The main backtracking encode algorithm

#include <stdio.h>

#include "iree/base/internal/math.h"
#include "iree/tokenizer/byte_level_tables.h"
#include "iree/tokenizer/model/bpe_internal.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

//===----------------------------------------------------------------------===//
// BPE Backtracking: Pair Validation
//===----------------------------------------------------------------------===//

// Maximum stack depth for iterative pair validation. Bounded by merge tree
// depth, which is at most max_token_length. 64 is generous for any real
// tokenizer (max_token_length rarely exceeds 128 bytes).
#define IREE_TOKENIZER_BPE_PAIR_VALIDATION_STACK_CAPACITY 64

// Validation frame stage: tracks where in the decomposition search we are.
typedef enum iree_tokenizer_bpe_validation_stage_e {
  // Initial stage: check blocking merge and base tokens.
  IREE_TOKENIZER_BPE_VALIDATION_STAGE_INIT = 0,
  // Trying decompositions of the first token to decompose (based on rank).
  IREE_TOKENIZER_BPE_VALIDATION_STAGE_FIRST,
  // Trying decompositions of the second token (fallback if first fails).
  IREE_TOKENIZER_BPE_VALIDATION_STAGE_SECOND,
} iree_tokenizer_bpe_validation_stage_t;

// Stack frame for iterative pair validation.
typedef struct iree_tokenizer_bpe_validation_frame_t {
  // Left token of the pair being validated.
  uint32_t token1;
  // Right token of the pair being validated.
  uint32_t token2;
  // Maximum rank allowed for blocking merges.
  uint32_t limit;
  // Current split position in decomposition search.
  // 0 = try split_table fast path first, 1+ = byte offset in text.
  uint16_t split_pos;
  uint8_t stage;  // iree_tokenizer_bpe_validation_stage_t
  // Which token to decompose: 1 = token1, 2 = token2.
  uint8_t side;
} iree_tokenizer_bpe_validation_frame_t;

// Determines whether two tokens can legally appear adjacent in a correct BPE
// encoding. Unlike a simple split_table-based approach, this tries ALL possible
// decomposition paths for each token. A pair is valid if there EXISTS at least
// one decomposition path where no cross-boundary merge should have fired.
//
// This is necessary because BPE can produce the same token via multiple merge
// paths, and the actual path taken depends on the global input context. For
// example, token "▁hell" can be formed via:
//   - ▁h + ell (if "ell" exists as a standalone token)
//   - ▁he + ll (if "ll" was formed by l+l merge)
//   - ▁hel + l
//   - ▁ + hell
//
// The split_table stores only the FIRST (lowest-rank) merge, but actual BPE
// might use a different path because earlier merges consumed intermediate
// tokens. We must check all paths to avoid false rejections.
//
// Implementation uses an explicit stack instead of recursion to:
//   - Prevent stack overflow on adversarial vocabularies with deep merge trees
//   - Return correct result (false) when stack exhausted instead of incorrectly
//     accepting unvalidatable pairs
//
// deferred_merge_rank: When suffix_blocked defers a token for better suffix
// merges, it reports the rank of the blocked token. Pairs whose merge has
// rank >= deferred_merge_rank are accepted (the merge was intentionally
// deferred, not missed). Pass 0 to disable this behavior.
static bool iree_tokenizer_bpe_is_valid_token_pair(
    const iree_tokenizer_bpe_model_t* model, uint32_t token1, uint32_t token2,
    uint32_t deferred_merge_rank) {
  const iree_tokenizer_vocab_t* vocab = model->vocab;
  const uint32_t* effective_rank = model->backtrack_tables.effective_rank;
  const iree_tokenizer_bpe_split_entry_t* split_table =
      model->backtrack_tables.split_table;

  iree_tokenizer_bpe_validation_frame_t
      stack[IREE_TOKENIZER_BPE_PAIR_VALIDATION_STACK_CAPACITY];
  iree_host_size_t stack_count = 0;

  // Push initial frame.
  stack[0] = (iree_tokenizer_bpe_validation_frame_t){
      .token1 = token1,
      .token2 = token2,
      .limit = UINT32_MAX,
      .split_pos = 0,
      .stage = IREE_TOKENIZER_BPE_VALIDATION_STAGE_INIT,
      .side = 0,
  };
  stack_count = 1;

  while (stack_count > 0) {
    iree_tokenizer_bpe_validation_frame_t* frame = &stack[stack_count - 1];

    if (frame->stage == IREE_TOKENIZER_BPE_VALIDATION_STAGE_INIT) {
      // Check whether a merge at this boundary should have been applied.
      iree_tokenizer_merge_hash_result_t merge =
          iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash,
                                                 (int32_t)frame->token1,
                                                 (int32_t)frame->token2);
      if (merge.result_id >= 0) {
        uint32_t combined_rank = effective_rank[(uint32_t)merge.result_id];
        // A merge is invalid if it should have fired (rank < limit) UNLESS
        // it was intentionally deferred for a better suffix merge. When
        // suffix_blocked triggers, it reports the rank of the blocked token;
        // merges at or above that rank were deferred, not missed.
        bool merge_should_have_fired = combined_rank < frame->limit;
        bool merge_was_deferred =
            deferred_merge_rank > 0 && combined_rank >= deferred_merge_rank;
        if (merge_should_have_fired && !merge_was_deferred) {
          // This merge should have fired → invalid pair. Pop and backtrack.
          --stack_count;
          continue;
        }
      }

      // Check if tokens are base tokens (split to self).
      bool token1_is_base =
          (split_table[frame->token1].right_id == frame->token1);
      bool token2_is_base =
          (split_table[frame->token2].left_id == frame->token2);

      if (token1_is_base && token2_is_base) {
        // Both sides are base tokens → valid pair! Propagate success up.
        return true;
      }

      // Need to split at least one token. Determine order based on rank.
      // Higher-ranked tokens (formed later in BPE) are split first.
      bool try_token1_first =
          effective_rank[frame->token1] > effective_rank[frame->token2];

      if (try_token1_first) {
        frame->side =
            token1_is_base ? 2 : 1;  // Skip to token2 if token1 is base.
      } else {
        frame->side =
            token2_is_base ? 1 : 2;  // Skip to token1 if token2 is base.
      }

      frame->stage = IREE_TOKENIZER_BPE_VALIDATION_STAGE_FIRST;
      frame->split_pos = 0;  // 0 = try split_table fast path first.
      continue;
    }

    // Get the token we're decomposing (based on which side).
    uint32_t decompose_token =
        (frame->side == 1) ? frame->token1 : frame->token2;

    bool found_decomposition = false;

    // Fast path: try split_table decomposition first (O(1), no hash lookups).
    // split_table stores the first reachable merge for each token. On the
    // first attempt (split_pos == 0), use it directly instead of scanning
    // all byte positions with hash lookups. This eliminates 3 hash lookups
    // per split position in the common case.
    if (frame->split_pos == 0) {
      frame->split_pos = 1;  // On backtrack, fall through to the full loop.
      iree_tokenizer_bpe_split_entry_t split = split_table[decompose_token];
      if (split.left_id != decompose_token) {
        // Non-base token: split_table provides (left_id, right_id) directly.
        // By construction, merge(left_id, right_id) == decompose_token.
        uint32_t new_limit;
        uint32_t new_token1, new_token2;
        if (frame->side == 1) {
          new_token1 = split.right_id;
          new_token2 = frame->token2;
          new_limit = effective_rank[frame->token1];
        } else {
          new_token1 = frame->token1;
          new_token2 = split.left_id;
          new_limit = effective_rank[frame->token2] + 1;
        }

        if (stack_count < IREE_TOKENIZER_BPE_PAIR_VALIDATION_STACK_CAPACITY) {
          stack[stack_count] = (iree_tokenizer_bpe_validation_frame_t){
              .token1 = new_token1,
              .token2 = new_token2,
              .limit = new_limit,
              .split_pos = 0,
              .stage = IREE_TOKENIZER_BPE_VALIDATION_STAGE_INIT,
              .side = 0,
          };
          ++stack_count;
          found_decomposition = true;
        } else {
          return false;
        }
      }
    }

    // Slow path: try all text split positions with hash lookups.
    // Only reached when the split_table fast path didn't apply (base token)
    // or failed (child validation rejected the split_table decomposition).
    if (!found_decomposition) {
      iree_string_view_t decompose_text =
          iree_tokenizer_vocab_token_text(vocab, (int32_t)decompose_token);

      // Compute the split_table's byte position so we can skip it (already
      // tried by the fast path). Recomputing is cheap: one array read + one
      // text length lookup, vs the 3 hash lookups we'd waste retrying it.
      iree_host_size_t table_split_pos = 0;
      {
        iree_tokenizer_bpe_split_entry_t split = split_table[decompose_token];
        if (split.left_id != decompose_token) {
          table_split_pos =
              iree_tokenizer_vocab_token_text(vocab, (int32_t)split.left_id)
                  .size;
        }
      }

      for (iree_host_size_t split_pos = frame->split_pos;
           split_pos < decompose_text.size; ++split_pos) {
        // Skip the position already tried by the fast path.
        if (split_pos == table_split_pos && table_split_pos > 0) continue;

        iree_string_view_t left_text =
            iree_make_string_view(decompose_text.data, split_pos);
        iree_string_view_t right_text = iree_make_string_view(
            decompose_text.data + split_pos, decompose_text.size - split_pos);

        // Look up both halves in vocab.
        int32_t left_id = iree_tokenizer_vocab_lookup(vocab, left_text);
        int32_t right_id = iree_tokenizer_vocab_lookup(vocab, right_text);
        if (left_id < 0 || right_id < 0) continue;

        // Check if there's a merge left + right → decompose_token.
        iree_tokenizer_merge_hash_result_t merge =
            iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, left_id,
                                                   right_id);
        if (merge.result_id != (int32_t)decompose_token) continue;

        // Valid decomposition found. Compute new limit and push child frame.
        uint32_t new_limit;
        uint32_t new_token1, new_token2;
        if (frame->side == 1) {
          new_token1 = (uint32_t)right_id;
          new_token2 = frame->token2;
          new_limit = effective_rank[frame->token1];
        } else {
          new_token1 = frame->token1;
          new_token2 = (uint32_t)left_id;
          new_limit = effective_rank[frame->token2] + 1;
        }

        // Save resume position for backtracking.
        frame->split_pos = (uint16_t)(split_pos + 1);
        found_decomposition = true;

        if (stack_count >= IREE_TOKENIZER_BPE_PAIR_VALIDATION_STACK_CAPACITY) {
          return false;
        }

        // Push child frame.
        stack[stack_count] = (iree_tokenizer_bpe_validation_frame_t){
            .token1 = new_token1,
            .token2 = new_token2,
            .limit = new_limit,
            .split_pos = 0,
            .stage = IREE_TOKENIZER_BPE_VALIDATION_STAGE_INIT,
            .side = 0,
        };
        ++stack_count;
        break;
      }
    }

    if (found_decomposition) {
      continue;  // Process the pushed child frame.
    }

    // No more decompositions for current side. Try the other side if in FIRST.
    if (frame->stage == IREE_TOKENIZER_BPE_VALIDATION_STAGE_FIRST) {
      bool token1_is_base =
          (split_table[frame->token1].right_id == frame->token1);
      bool token2_is_base =
          (split_table[frame->token2].left_id == frame->token2);

      // Switch to the other side.
      uint8_t other_side = (frame->side == 1) ? 2 : 1;
      bool other_is_base = (other_side == 1) ? token1_is_base : token2_is_base;

      if (!other_is_base) {
        frame->stage = IREE_TOKENIZER_BPE_VALIDATION_STAGE_SECOND;
        frame->side = other_side;
        frame->split_pos = 0;  // 0 = try split_table fast path first.
        continue;
      }
    }

    // All decompositions exhausted for this frame. Pop and backtrack.
    --stack_count;
  }

  // Stack empty with no valid path found.
  return false;
}

// Computes how many raw input bytes a token covers starting at a given
// position. For non-ByteLevel mode: equals the token's text length. For
// ByteLevel mode: walks raw bytes, accumulating their UTF-8 byte lengths, until
// reaching the token's text length.
static iree_host_size_t iree_tokenizer_bpe_token_raw_length(
    const iree_tokenizer_bpe_model_t* model, uint32_t token_id,
    const uint8_t* segment_data, iree_host_size_t start_position) {
  iree_string_view_t token_text =
      iree_tokenizer_vocab_token_text(model->vocab, (int32_t)token_id);

  if (!iree_all_bits_set(model->flags,
                         IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT)) {
    return token_text.size;
  }

  // ByteLevel: each raw byte produces 1-2 UTF-8 bytes. Walk raw bytes until
  // accumulated UTF-8 length reaches the token text length.
  // Printable ASCII (0x21-0x7E) maps identity (1 UTF-8 byte per raw byte);
  // all other bytes map to codepoints >= 0x80 (2 UTF-8 bytes per raw byte).
  iree_host_size_t utf8_accumulated = 0;
  iree_host_size_t raw_consumed = 0;
  while (utf8_accumulated < token_text.size) {
    uint8_t byte = segment_data[start_position + raw_consumed];
    utf8_accumulated += (byte >= 0x21 && byte <= 0x7E) ? 1 : 2;
    raw_consumed++;
  }
  return raw_consumed;
}

// Looks up the single-character token for a raw byte.
// Returns the token ID, or -1 if no single-char token exists.
// Note: byte_to_token is precomputed at model init with ByteLevel handling.
static inline int32_t iree_tokenizer_bpe_single_char_token(
    const iree_tokenizer_bpe_model_t* model, uint8_t raw_byte) {
  return model->byte_to_token[raw_byte];
}

// Finds the first (shortest) valid token at a given position in the input
// by walking the trie. Returns -1 if no valid token is found within
// max_bytes or before the trie walk fails.
static int32_t iree_tokenizer_bpe_find_first_token_at(
    const iree_tokenizer_bpe_model_t* model, const uint8_t* data,
    iree_host_size_t size, iree_host_size_t max_bytes,
    iree_host_size_t* out_length) {
  if (size == 0) return -1;

  const bool byte_level =
      iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);

  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  iree_host_size_t limit = size < max_bytes ? size : max_bytes;
  for (iree_host_size_t i = 0; i < limit; ++i) {
    if (!iree_tokenizer_bpe_trie_advance_byte(&cursor, data[i], byte_level)) {
      break;
    }
    int32_t token = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (token >= 0) {
      if (out_length) *out_length = i + 1;
      return token;
    }
  }
  return -1;
}

// Looks up the base token at a given byte position. Tries single-char lookup
// first (always succeeds for byte-level tokenizers). Falls back to trie lookup
// for SentencePiece-style tokenizers with multi-byte base tokens.
// Returns -1 if no token found. Sets *out_length if non-NULL.
static int32_t iree_tokenizer_bpe_token_at_position(
    const iree_tokenizer_bpe_model_t* model, const uint8_t* data,
    iree_host_size_t size, iree_host_size_t* out_length) {
  int32_t token = iree_tokenizer_bpe_single_char_token(model, data[0]);
  if (token >= 0) {
    if (out_length) *out_length = 1;
    return token;
  }
  return iree_tokenizer_bpe_find_first_token_at(model, data, size, 8,
                                                out_length);
}

// Checks whether a token at a given position is consumed by a rightward merge
// at rank lower than |max_rank|. This means merge(token, following) exists and
// fires before any merge at |max_rank| would.
//
// To verify the rightward merge actually fires, also checks that the following
// token isn't itself consumed by an even lower-rank merge. This handles chains
// like: merge(c,h) would consume c, but merge(h,X) at lower rank consumes h
// first — so merge(c,h) can't fire and c survives.
static bool iree_tokenizer_bpe_is_token_consumed_rightward(
    const iree_tokenizer_bpe_model_t* model, int32_t token,
    const uint8_t* remaining_data, iree_host_size_t remaining_size,
    iree_host_size_t token_end, uint32_t max_rank) {
  if (token_end >= remaining_size) return false;
  iree_host_size_t following_length = 1;
  int32_t following_token = iree_tokenizer_bpe_token_at_position(
      model, remaining_data + token_end, remaining_size - token_end,
      &following_length);
  if (following_token < 0) return false;

  iree_tokenizer_merge_hash_result_t merge =
      iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, token,
                                             following_token);
  if (!iree_tokenizer_merge_hash_result_is_valid(merge) ||
      merge.rank >= max_rank) {
    return false;
  }

  // merge(token, following) would fire. Verify following is available: if
  // merge(following, next_after) fires at even lower rank, following is
  // consumed and merge(token, following) can't fire.
  iree_host_size_t next_after_end = token_end + following_length;
  if (next_after_end < remaining_size) {
    int32_t next_after_token = iree_tokenizer_bpe_token_at_position(
        model, remaining_data + next_after_end, remaining_size - next_after_end,
        NULL);
    if (next_after_token >= 0) {
      iree_tokenizer_merge_hash_result_t following_merge =
          iree_tokenizer_vocab_merge_hash_lookup(
              model->merge_hash, following_token, next_after_token);
      if (iree_tokenizer_merge_hash_result_is_valid(following_merge) &&
          following_merge.rank < merge.rank) {
        return false;  // Following consumed first, token survives.
      }
    }
  }
  return true;
}

// Checks if a suffix merge is preempted by lower-rank merges in the following
// input. Returns true if the suffix merge would NOT actually fire because the
// prefix token would be consumed first.
//
// Two preemption paths:
//   Direct: merge(prefix, next) at rank < suffix_merge_rank, where the next
//     token is available (not consumed by a lower-rank rightward merge).
//   Compound: merge(prefix, compound) where compound is built by progressively
//     merging characters: next+third->c1, c1+fourth->c2, etc. This handles
//     cases where the direct merge can't fire but the prefix is still consumed
//     via a deeper compound (e.g., merge(l, ish) when i+s fires first).
//
// For non-byte-level tokenizers (e.g., SentencePiece), uses trie lookup to find
// tokens when single-char lookup fails. This handles multi-byte base tokens
// like metaspace (E2 96 81).
static bool iree_tokenizer_bpe_is_suffix_merge_preempted(
    const iree_tokenizer_bpe_model_t* model, int32_t prefix_token,
    uint32_t suffix_merge_rank, const uint8_t* remaining_data,
    iree_host_size_t remaining_size, iree_host_size_t prefix_end) {
  if (prefix_end >= remaining_size) return false;

  iree_host_size_t next_length = 1;
  int32_t next_token = iree_tokenizer_bpe_token_at_position(
      model, remaining_data + prefix_end, remaining_size - prefix_end,
      &next_length);
  if (next_token < 0) return false;

  // Direct preemption: merge(prefix, next) at rank < suffix_merge_rank.
  // Only valid if next is actually available (not consumed rightward).
  iree_tokenizer_merge_hash_result_t prefix_next =
      iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, prefix_token,
                                             next_token);
  if (iree_tokenizer_merge_hash_result_is_valid(prefix_next) &&
      prefix_next.rank < suffix_merge_rank) {
    if (!iree_tokenizer_bpe_is_token_consumed_rightward(
            model, next_token, remaining_data, remaining_size,
            prefix_end + next_length, prefix_next.rank)) {
      return true;
    }
  }

  // Compound preemption: build progressively longer compounds from the
  // characters following the prefix, checking merge(prefix, compound) at each
  // step. This catches cases where the direct merge can't fire but the prefix
  // merges with a multi-character compound that forms at lower rank.
  //
  // Example (Whisper "establish"):
  //   prefix=l, next=i, merge(l,i) can't fire (i consumed by i+s at rank 15)
  //   compound: i+s->is (rank 15), is+h->ish (rank 486)
  //   merge(l, ish) at rank 1677 < suffix_merge_rank -> preempted
  iree_host_size_t compound_end = prefix_end + next_length;
  iree_host_size_t extension_length = 1;
  int32_t extension_token = -1;
  if (compound_end < remaining_size) {
    extension_token = iree_tokenizer_bpe_token_at_position(
        model, remaining_data + compound_end, remaining_size - compound_end,
        &extension_length);
  }
  if (extension_token < 0) return false;

  iree_tokenizer_merge_hash_result_t compound_merge =
      iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, next_token,
                                             extension_token);
  if (!iree_tokenizer_merge_hash_result_is_valid(compound_merge) ||
      compound_merge.rank >= suffix_merge_rank) {
    return false;
  }

  // Build the compound progressively, checking at each level.
  int32_t compound_token = compound_merge.result_id;
  compound_end += extension_length;
  for (int depth = 0; depth < 4; ++depth) {
    iree_tokenizer_merge_hash_result_t prefix_compound =
        iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, prefix_token,
                                               compound_token);
    if (iree_tokenizer_merge_hash_result_is_valid(prefix_compound) &&
        prefix_compound.rank < suffix_merge_rank) {
      return true;
    }

    // Extend compound by one more character.
    if (compound_end >= remaining_size) break;
    extension_token = iree_tokenizer_bpe_token_at_position(
        model, remaining_data + compound_end, remaining_size - compound_end,
        &extension_length);
    if (extension_token < 0) break;

    compound_merge = iree_tokenizer_vocab_merge_hash_lookup(
        model->merge_hash, compound_token, extension_token);
    if (!iree_tokenizer_merge_hash_result_is_valid(compound_merge) ||
        compound_merge.rank >= suffix_merge_rank) {
      break;
    }
    compound_token = compound_merge.result_id;
    compound_end += extension_length;
  }
  return false;
}

// Checks if a candidate token is "suffix-blocked" - meaning a proper suffix of
// the token can merge with a prefix of the remaining input at a lower rank than
// the token's effective rank. If so, choosing this token would prevent a better
// merge from happening.
//
// Example: For input "oising" with token "ois" (effective_rank ~10669):
//   - "is" is a suffix of "ois" (from split(ois) = (o, is))
//   - "ing" is a prefix of the remaining input "ing"
//   - merge(is, ing) has rank 1454 < 10669
//   - So "ois" is suffix-blocked: we should choose "o" + "ising" instead.
//
// For models with end_of_word_suffix (e.g., CLIP's "</w>"), also checks if the
// token's suffix can merge with a prefix of (remaining + end_of_word_suffix).
// This handles cases like CLIP's 10-emoji input where token 25159 (🎉🎉) at
// position 24 would be blocked because:
//   - suffix(25159) = 4310 (🎉)
//   - remaining + suffix = 🎉🎉</w> (8 bytes + 4 bytes)
//   - prefix 🎉🎉</w> = token 13450
//   - merge(4310, 13450) = 20828 at rank 20316 < 25159's rank 24647
// This allows the algorithm to choose 4310 at position 24, leading to the
// better tokenization [25159, 25159, 25159, 4310, 20828].
//
// This check walks the FULL rightmost path of the token's merge tree, checking
// each suffix against ALL prefix tokens of the remaining input.
//
// When blocked, out_blocking_rank is set to the rank of the blocking merge.
// This allows pair validation to accept pairs whose merge rank >=
// blocking_rank, since those merges were intentionally deferred for better
// suffix merges.
static bool iree_tokenizer_bpe_is_suffix_blocked(
    const iree_tokenizer_bpe_model_t* model, int32_t token,
    const uint8_t* remaining_data, iree_host_size_t remaining_size,
    const char* suffix_data, iree_host_size_t suffix_length,
    uint32_t* out_blocking_rank) {
  *out_blocking_rank = 0;
  if (remaining_size == 0 && suffix_length == 0) return false;

  const iree_tokenizer_bpe_split_entry_t* split_table =
      model->backtrack_tables.split_table;
  const uint32_t* effective_rank = model->backtrack_tables.effective_rank;
  const bool byte_level =
      iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);

  uint32_t token_rank = effective_rank[(uint32_t)token];
  if (token_rank <= 1) return false;  // Base token, no suffix to check.

  // Collect all rightmost suffixes with their consumption ranks.
  // A suffix is "consumed" when its parent forms - if a potential suffix merge
  // has rank >= consumption rank, the suffix is already gone.
  //
  // Important: Only collect suffixes from REACHABLE decompositions. The
  // split_table records the first merge that produces a token, but that merge
  // path may be blocked by lower-rank boundary merges. If a decomposition is
  // not reachable, the suffix from that decomposition doesn't exist in proper
  // BPE output.
  //
  // Example: ▁hell has split_table entry (▁h, ell), but this decomposition is
  // blocked by h+e at the boundary. The actual BPE path is ▁he+ll, so the
  // suffix is 'll', not 'ell'. Without this check, we would incorrectly block
  // ▁hell because ell+o can merge at lower rank than ▁hell.
  uint32_t suffixes[32];
  uint32_t suffix_consumed_at[32];
  iree_host_size_t suffix_count = 0;

  for (uint32_t current = (uint32_t)token;
       suffix_count < 32 && split_table[current].left_id != current;
       current = split_table[current].right_id) {
    // Check if this decomposition is actually reachable via BPE. If the
    // split_table decomposition is blocked by a boundary merge, skip this
    // suffix - it won't exist in the actual BPE token sequence.
    uint32_t left = split_table[current].left_id;
    uint32_t right = split_table[current].right_id;
    bool reachable = iree_tokenizer_bpe_is_decomposition_reachable(
        model, left, right, current);
    if (!reachable) {
      // This decomposition is blocked. The actual BPE path uses a different
      // decomposition, so we can't rely on this suffix chain. Stop collecting
      // suffixes here - the actual suffixes depend on the alternate path which
      // we don't have easy access to.
      break;
    }
    suffixes[suffix_count] = right;
    suffix_consumed_at[suffix_count] = effective_rank[current];
    suffix_count++;
  }
  IREE_ASSERT(suffix_count < IREE_ARRAYSIZE(suffixes),
              "suffix chain depth exceeds capacity");
  if (suffix_count == 0) return false;

  // Walk the trie to enumerate all prefix tokens of the remaining input.
  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  // Phase 1: Walk remaining input bytes.
  iree_host_size_t bytes_consumed = 0;
  for (iree_host_size_t i = 0; i < remaining_size; ++i) {
    if (!iree_tokenizer_bpe_trie_advance_byte(&cursor, remaining_data[i],
                                              byte_level)) {
      // Trie walk failed - can't reach suffix bytes.
      return false;
    }
    bytes_consumed = i + 1;

    int32_t prefix_token = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (prefix_token < 0) continue;

    // Check each suffix against this prefix.
    for (iree_host_size_t s = 0; s < suffix_count; ++s) {
      iree_tokenizer_merge_hash_result_t merge =
          iree_tokenizer_vocab_merge_hash_lookup(
              model->merge_hash, (int32_t)suffixes[s], prefix_token);
      if (!iree_tokenizer_merge_hash_result_is_valid(merge)) continue;

      // Skip if merge rank is too high to matter.
      uint32_t merge_effective_rank = merge.rank + 1;
      if (merge_effective_rank >= token_rank) continue;

      // Skip if the suffix is consumed before this merge could fire.
      if (merge_effective_rank >= suffix_consumed_at[s]) continue;

      // Skip when suffix == prefix (repeating pattern case).
      // When the suffix token equals the prefix token, we have a repeating
      // pattern like consecutive metaspaces. In this case, any shorter token
      // would also face suffix blocking by the same pattern, creating an
      // infinite loop. The suffix blocking optimization doesn't help here
      // because there's no "better" tokenization to defer to.
      if ((int32_t)suffixes[s] == prefix_token) continue;

      // Check if the prefix would be consumed by lower-rank merges first.
      if (iree_tokenizer_bpe_is_suffix_merge_preempted(
              model, prefix_token, merge.rank, remaining_data, remaining_size,
              i + 1)) {
        continue;
      }

      // Verify the prefix token can actually form at this position. A
      // multi-character prefix requires its internal merges to complete
      // before external boundary merges consume its edge characters. If
      // the rightmost base of the prefix merges with the following
      // character at a rank lower than the prefix's internal consumption
      // of that base, the prefix cannot form here.
      if (effective_rank[(uint32_t)prefix_token] > 1 &&
          i + 1 < remaining_size) {
        uint32_t rightmost_base = iree_tokenizer_bpe_rightmost_base_token(
            model, (uint32_t)prefix_token);
        uint32_t right_consumed =
            iree_tokenizer_bpe_right_boundary_consumed_rank(
                model, (uint32_t)prefix_token);
        if (iree_tokenizer_bpe_is_token_consumed_rightward(
                model, (int32_t)rightmost_base, remaining_data, remaining_size,
                i + 1, right_consumed)) {
          continue;  // Prefix can't form at this position.
        }
      }

      // Found a valid blocking merge. Report its rank so pair validation
      // can accept pairs whose merge would fire at or above this rank
      // (since we're intentionally deferring for this better suffix merge).
      *out_blocking_rank = token_rank;
      return true;
    }
  }

  // Phase 2: Continue with end_of_word_suffix bytes to find suffixed tokens.
  // Only proceed if we consumed all remaining bytes (otherwise trie walk broke
  // earlier and we can't reach suffix bytes).
  if (bytes_consumed != remaining_size || suffix_length == 0) {
    return false;
  }

  for (iree_host_size_t i = 0; i < suffix_length; ++i) {
    // Suffix bytes are raw (not ByteLevel encoded).
    if (!iree_tokenizer_trie_cursor_advance(&cursor, (uint8_t)suffix_data[i])) {
      return false;  // No more prefixes possible.
    }

    int32_t prefix_token = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (prefix_token < 0) continue;

    // Check each token suffix against this suffixed prefix token.
    for (iree_host_size_t s = 0; s < suffix_count; ++s) {
      iree_tokenizer_merge_hash_result_t merge =
          iree_tokenizer_vocab_merge_hash_lookup(
              model->merge_hash, (int32_t)suffixes[s], prefix_token);
      if (!iree_tokenizer_merge_hash_result_is_valid(merge)) continue;

      // Skip if merge rank is too high to matter.
      uint32_t merge_effective_rank = merge.rank + 1;
      if (merge_effective_rank >= token_rank) continue;

      // Skip if the token suffix is consumed before this merge could fire.
      if (merge_effective_rank >= suffix_consumed_at[s]) continue;

      // Skip when suffix == prefix (repeating pattern case).
      if ((int32_t)suffixes[s] == prefix_token) continue;

      // Suffixed tokens consume all remaining input + suffix bytes, so no
      // preemption check is needed. The prefix_token represents the complete
      // remaining segment (remaining_data + partial suffix) and nothing
      // follows it that could preempt the merge.
      *out_blocking_rank = token_rank;
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// BPE Backtracking: Core Algorithm
//===----------------------------------------------------------------------===//

// Runs the O(n) backtracking BPE encode algorithm on a segment.
// Populates the backtrack_stack with the resulting token sequence.
// The caller emits tokens from the stack after this function returns.
//
// Algorithm: greedy longest-match with pair validation and backtracking.
// For each position, find the longest trie match. If the pair (previous,
// current) is valid and the end position is reachable, accept. Otherwise try
// shorter prefix tokens. If no prefix works, backtrack to the previous token
// and try its shorter prefixes.
//
// suffix_data/suffix_length: End-of-word suffix (e.g., "</w>") to consider in
// suffix_blocked checks. When a token's suffix can merge with a prefix of
// (remaining + end_of_word_suffix) at lower rank, the token is blocked.
void iree_tokenizer_bpe_backtrack_encode(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    const uint8_t* segment_data, iree_host_size_t segment_size,
    const char* suffix_data, iree_host_size_t suffix_length) {
  iree_tokenizer_bpe_backtrack_entry_t* stack =
      iree_tokenizer_bpe_state_backtrack_stack(state, model);
  uint64_t* bitfield =
      iree_tokenizer_bpe_state_backtrack_bitfield(state, model);

  // Reset only the bitfield words dirtied by the previous segment's
  // backtracking. On the first call, dirty_mask is pre-set to all-ones by
  // state_initialize, so this performs the equivalent of a full init.
  // On subsequent calls with no backtracking, dirty_mask is 0 and this
  // loop body executes zero times.
  uint64_t dirty = state->backtrack.dirty_mask;
  while (dirty) {
    iree_host_size_t word_index =
        (iree_host_size_t)iree_math_count_trailing_zeros_u64(dirty);
    bitfield[word_index] = UINT64_MAX;
    dirty &= dirty - 1;  // Clear lowest set bit.
  }
  state->backtrack.dirty_mask = 0;

  iree_host_size_t stack_count = 0;
  iree_host_size_t position = 0;

  // Tracks the minimum position above which tokens may be popped during
  // backtracking. Byte-fallback tokens are unconditional (they represent the
  // only possible tokenization for that byte) and must never be reverted.
  // When byte-fallback fires at position P, committed_position advances to
  // P+1, preventing any pop from reverting past that point.
  iree_host_size_t committed_position = 0;

  // Get initial longest match.
  int32_t next_token = -1;
  iree_host_size_t next_token_raw_length = 0;
  iree_tokenizer_bpe_backtrack_longest_match(
      model, segment_data, segment_size, &next_token, &next_token_raw_length);

  while (position < segment_size) {
    // If no trie match at this position, fall back to byte-level tokens.
    if (next_token < 0) {
      int32_t byte_token = model->byte_to_token[segment_data[position]];
      if (IREE_UNLIKELY(byte_token < 0)) {
        byte_token = iree_tokenizer_bpe_handle_unknown_byte(
            model, segment_data[position],
            stack_count > 0 ? stack[stack_count - 1].token_id : -1);
      }
      if (byte_token < 0) {
        // FUSE_UNK: byte fused with previous UNK on stack. Skip to next byte.
        // Advance committed_position: the fused byte is definitively handled
        // and must not be re-visited by backtracking. Without this, the
        // backtracking loop can pop past fused positions and re-accept the
        // same token sequence indefinitely (infinite loop).
        position++;
        committed_position = position;
        if (position < segment_size) {
          iree_tokenizer_bpe_backtrack_longest_match(
              model, segment_data + position, segment_size - position,
              &next_token, &next_token_raw_length);
        }
        continue;
      }
      IREE_ASSERT(stack_count < model->backtrack_stack_capacity);
      iree_tokenizer_bpe_backtrack_entry_set(&stack[stack_count], byte_token,
                                             (uint32_t)position, 0);
      stack_count++;
      position++;
      committed_position = position;
      if (position < segment_size) {
        iree_tokenizer_bpe_backtrack_longest_match(
            model, segment_data + position, segment_size - position,
            &next_token, &next_token_raw_length);
      }
      continue;
    }

    int32_t token = next_token;
    iree_host_size_t token_raw_length = next_token_raw_length;
    int32_t last_token = stack_count > 0 ? stack[stack_count - 1].token_id : -1;
    bool first_token = last_token < 0;

    // Track the deferred merge rank across prefix attempts at this position.
    // When a longer token is suffix_blocked, we try shorter prefixes. The
    // blocking rank is accumulated and stored with whichever token is accepted,
    // so that pair validation at the NEXT position knows which merges were
    // intentionally deferred. We track the maximum blocking rank seen across
    // all prefix attempts at this position.
    uint32_t position_deferred_merge_rank = 0;

    // For pair validation, use the deferred_merge_rank stored with the previous
    // token. This is the rank that was active when the previous token was
    // pushed - it tells us which merges at this boundary were intentionally
    // deferred. For example, if 'lish' was suffix_blocked and we accepted 'li',
    // then when checking (li, sh), we need the blocking rank from 'lish' to
    // know that li+sh→lish was intentionally skipped.
    uint32_t pair_deferred_merge_rank =
        (stack_count > 0) ? iree_tokenizer_bpe_backtrack_entry_deferred_rank(
                                &stack[stack_count - 1])
                          : 0;

    // Check if remaining_segment + suffix matches a vocabulary token directly.
    // This handles cases like "lines" (not in vocab) where "lines</w>" IS in
    // vocabulary. Without this check, suffix_blocking would reject individual
    // tokens (e.g., "line" blocked because e+s</w> merges), but the explicit
    // suffixed token "lines</w>" should be used directly when available.
    //
    // Only use this shortcut when the normal longest-match token does not
    // cover the entire remaining segment. If next_token covers the
    // entire remaining, we should go through normal BPE suffix_blocked checks
    // to ensure correct merge ordering. This prevents short-circuiting when
    // BPE would produce a different (correct) tokenization.
    //
    // Example: "bbbb" with merges b+b->bb, bb+b->bbb
    // - At position 0, next_token="bbb" (3 bytes) doesn't cover all 4 bytes
    // - Normal BPE with suffix_blocked should reject "bbb" and give [bb, bb]
    // - We should NOT short-circuit with suffix_lookup here
    //
    // Example: "lines" with no "lines" token but "lines</w>" exists
    // - At position 0 (after "new"), next_token="line" (4 bytes)
    // - remaining="lines" (5 bytes), so next_token doesn't cover all
    // - suffix_lookup("lines</w>") finds token 3418, which we use directly
    if (suffix_length > 0 && next_token_raw_length < segment_size - position) {
      iree_string_view_t remaining = iree_make_string_view(
          (const char*)(segment_data + position), segment_size - position);
      iree_string_view_t suffix =
          iree_make_string_view(suffix_data, suffix_length);
      int32_t suffixed_token_id = -1;
      iree_host_size_t suffixed_length = 0;
      iree_tokenizer_bpe_trie_longest_match_with_suffix(
          model, remaining, suffix, &suffixed_token_id, &suffixed_length);
      if (suffixed_token_id >= 0 &&
          suffixed_length == (segment_size - position) + suffix_length) {
        // The entire remaining segment + suffix matches a vocabulary token.
        // Check pair validity with predecessor.
        //
        // We skip the reachability check here. The reachability check is
        // designed for the incremental BPE merge algorithm, not for direct
        // vocabulary lookups. When we find a suffixed token like "lines</w>"
        // via trie lookup, we're matching a complete vocabulary entry directly,
        // not building it through BPE merges. The token is valid because it
        // exists in the vocabulary and matches the input exactly.
        bool suffixed_pair_ok =
            first_token ||
            iree_tokenizer_bpe_is_valid_token_pair(model, (uint32_t)last_token,
                                                   (uint32_t)suffixed_token_id,
                                                   pair_deferred_merge_rank);
        if (suffixed_pair_ok) {
          // Use the suffixed token directly. This bypasses the normal
          // backtracking loop since we found the optimal final token.
          IREE_ASSERT(stack_count < model->backtrack_stack_capacity);
          iree_tokenizer_bpe_backtrack_entry_set(
              &stack[stack_count], suffixed_token_id, (uint32_t)position, 0);
          stack_count++;
          position = segment_size;
          next_token = -1;
          continue;  // Jump back to outer while loop - segment complete.
        }
      }
    }

    for (;;) {
      iree_host_size_t end_position = position + token_raw_length;

      //===------------------------------------------------------------------===//
      // Token Acceptance Decision
      //
      // A candidate token is accepted only if ALL four conditions pass:
      //   1. bitfield_ok: End position is reachable (not ruled out by prior
      //      backtracking).
      //   2. !token_unreachable: The token's BPE merge path is not blocked by
      //      lower-rank internal merges.
      //   3. pair_ok: This token can legally follow the previous token (no
      //      missed boundary merges).
      //   4. !suffix_blocked: No better tokenization exists via suffix merges.
      //
      // When suffix_blocked, the blocking rank is recorded so pair validation
      // at the NEXT position knows which merges were intentionally deferred.
      //===------------------------------------------------------------------===//

      // Check 1: Bitfield reachability - was this position ruled out?
      bool bitfield_ok =
          iree_tokenizer_bpe_bitfield_is_set(bitfield, end_position);

      // Check 2: Token internal reachability - can this token actually form?
      // Multi-byte merged tokens (effective_rank > 1) need validation to ensure
      // the token's merge path is not blocked by lower-rank merges.
      // Example: "hello" with merges "l l"(1), "he l"(9) - the "l l" merge
      // fires first, blocking the path to "hello" as a single token.
      bool token_unreachable =
          token_raw_length > 1 &&
          model->backtrack_tables.effective_rank[(uint32_t)token] > 1 &&
          !iree_any_bit_set(
              model->backtrack_tables.token_reachable[(uint32_t)token / 64],
              1ull << ((uint32_t)token % 64));

      // Check 3: Suffix blocking - would a better tokenization exist?
      // Prevents choosing "ois" when "is+ing->ising" has higher priority.
      // When blocked, the rank is recorded for pair validation at next
      // position.
      uint32_t blocking_rank = 0;
      bool suffix_blocked = iree_tokenizer_bpe_is_suffix_blocked(
          model, token, segment_data + end_position,
          segment_size - end_position, suffix_data, suffix_length,
          &blocking_rank);
      if (suffix_blocked) {
        position_deferred_merge_rank = blocking_rank;
      }

      // Check 4: Pair validity - can this token follow the previous one?
      bool pair_ok =
          first_token || iree_tokenizer_bpe_is_valid_token_pair(
                             model, (uint32_t)last_token, (uint32_t)token,
                             pair_deferred_merge_rank);

      // Final decision: accept only if all checks pass.
      bool accept =
          bitfield_ok && !token_unreachable && pair_ok && !suffix_blocked;

      if (accept) {
        // Push token onto stack and advance position.
        IREE_ASSERT(stack_count < model->backtrack_stack_capacity);
        iree_tokenizer_bpe_backtrack_entry_set(&stack[stack_count], token,
                                               (uint32_t)position,
                                               position_deferred_merge_rank);
        stack_count++;
        position = end_position;

        // Find next longest match from new position.
        if (position < segment_size) {
          iree_tokenizer_bpe_backtrack_longest_match(
              model, segment_data + position, segment_size - position,
              &next_token, &next_token_raw_length);
        } else {
          next_token = -1;
        }
        break;

      } else {
        // Try a shorter prefix token.
        uint32_t prefix =
            model->backtrack_tables.next_prefix_match[(uint32_t)token];
        if (prefix != UINT32_MAX) {
          token = (int32_t)prefix;
          token_raw_length = iree_tokenizer_bpe_token_raw_length(
              model, prefix, segment_data, position);
          continue;
        }

        // No shorter prefix available — backtrack.
        iree_tokenizer_bpe_bitfield_clear(
            bitfield, &state->backtrack.dirty_mask, position);

        if (stack_count == 0 ||
            iree_tokenizer_bpe_backtrack_entry_start_byte(
                &stack[stack_count - 1]) < committed_position) {
          // Either the stack is empty or the top entry is a committed
          // byte-fallback that must not be reverted. Use byte-level fallback
          // at the current position and advance unconditionally.
          int32_t byte_token = model->byte_to_token[segment_data[position]];
          if (byte_token < 0) {
            byte_token = iree_tokenizer_bpe_handle_unknown_byte(
                model, segment_data[position],
                stack_count > 0 ? stack[stack_count - 1].token_id : -1);
          }
          if (byte_token < 0) {
            // FUSE_UNK: byte fused with previous UNK on stack. Skip to next
            // byte.
            position++;
            committed_position = position;
            if (position < segment_size) {
              iree_tokenizer_bpe_backtrack_longest_match(
                  model, segment_data + position, segment_size - position,
                  &next_token, &next_token_raw_length);
            } else {
              next_token = -1;
            }
            break;
          }
          IREE_ASSERT(stack_count < model->backtrack_stack_capacity);
          iree_tokenizer_bpe_backtrack_entry_set(
              &stack[stack_count], byte_token, (uint32_t)position, 0);
          stack_count++;
          position++;
          committed_position = position;
          if (position < segment_size) {
            iree_tokenizer_bpe_backtrack_longest_match(
                model, segment_data + position, segment_size - position,
                &next_token, &next_token_raw_length);
          } else {
            next_token = -1;
          }
          break;
        }

        // Pop the previous token and retry it.
        stack_count--;
        position =
            iree_tokenizer_bpe_backtrack_entry_start_byte(&stack[stack_count]);
        next_token = stack[stack_count].token_id;
        next_token_raw_length = iree_tokenizer_bpe_token_raw_length(
            model, (uint32_t)next_token, segment_data, position);
        break;
      }
    }
  }

  state->backtrack.stack_count = stack_count;
  state->backtrack.emit_index = 0;
}
