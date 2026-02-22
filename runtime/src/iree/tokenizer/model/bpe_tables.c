// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE table construction: builds split_table, effective_rank, and
// token_reachable bitmap used by the backtracking algorithm.

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/tokenizer/model/bpe_internal.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

//===----------------------------------------------------------------------===//
// BPE Backtracking: First Token Reachability
//===----------------------------------------------------------------------===//

// Returns the rightmost leaf (base) token of the split tree rooted at token.
// For a base token (split_table[token] = {token, token}), returns itself.
// Used to find the boundary token when checking for blocking merges.
uint32_t iree_tokenizer_bpe_rightmost_base_token(
    const iree_tokenizer_bpe_model_t* model, uint32_t token) {
  const iree_tokenizer_bpe_split_entry_t* split_table =
      model->backtrack_tables.split_table;
  while (split_table[token].right_id != token) {
    token = split_table[token].right_id;
  }
  return token;
}

// Returns the leftmost leaf (base) token of the split tree rooted at token.
static uint32_t iree_tokenizer_bpe_leftmost_base_token(
    const iree_tokenizer_bpe_model_t* model, uint32_t token) {
  const iree_tokenizer_bpe_split_entry_t* split_table =
      model->backtrack_tables.split_table;
  while (split_table[token].left_id != token) {
    token = split_table[token].left_id;
  }
  return token;
}

// Returns the effective_rank when the rightmost boundary base token is first
// consumed by a merge within the subtree rooted at |token|.
//
// This walks down the right spine of the split tree until it finds where the
// rightmost base token is first merged with its left sibling. This rank is
// when the boundary token becomes unavailable for external merges.
//
// Example: For Ġfib = merge(Ġf, ib) where ib = merge(i, b):
//   - Rightmost base token is 'b'
//   - 'b' is consumed when 'ib' forms (not when 'Ġfib' forms)
//   - Returns effective_rank[ib] = 428
uint32_t iree_tokenizer_bpe_right_boundary_consumed_rank(
    const iree_tokenizer_bpe_model_t* model, uint32_t token) {
  const iree_tokenizer_bpe_split_entry_t* split_table =
      model->backtrack_tables.split_table;
  const uint32_t* effective_rank = model->backtrack_tables.effective_rank;

  // Walk down the right spine until we find where the rightmost base token
  // is consumed (i.e., where the right child is a base token).
  while (split_table[token].left_id != token) {
    uint32_t right = split_table[token].right_id;
    if (split_table[right].left_id == right) {
      // Right child is a base token - it's consumed at this token's merge.
      return effective_rank[token];
    }
    // Recurse into right subtree.
    token = right;
  }
  // Token is a base token itself - it's never consumed within a subtree.
  return UINT32_MAX;
}

// Returns the effective_rank when the leftmost boundary base token is first
// consumed by a merge within the subtree rooted at |token|.
//
// Symmetric to right_boundary_consumed_rank, but walks the left spine.
static uint32_t iree_tokenizer_bpe_left_boundary_consumed_rank(
    const iree_tokenizer_bpe_model_t* model, uint32_t token) {
  const iree_tokenizer_bpe_split_entry_t* split_table =
      model->backtrack_tables.split_table;
  const uint32_t* effective_rank = model->backtrack_tables.effective_rank;

  // Walk down the left spine until we find where the leftmost base token
  // is consumed (i.e., where the left child is a base token).
  while (split_table[token].left_id != token) {
    uint32_t left = split_table[token].left_id;
    if (split_table[left].left_id == left) {
      // Left child is a base token - it's consumed at this token's merge.
      return effective_rank[token];
    }
    // Recurse into left subtree.
    token = left;
  }
  // Token is a base token itself - it's never consumed within a subtree.
  return UINT32_MAX;
}

// Checks if a token is reachable from its input bytes via proper BPE merge
// ordering. A token is NOT reachable if a lower-rank merge would fire at an
// internal split boundary, blocking the token's merge path.
//
// Example: "hello" with merges "h e"(0), "l l"(1), "he l"(9), "ll o"(10),
// "hel lo"(13). Token "hello" splits into [hel, lo]. At the boundary between
// "hel" and "lo", the base tokens are 'l' and 'l'. Merge "l l" at rank 1 would
// fire before "hel lo" at rank 13, blocking the path to "hello".
//
// This check is only needed for first tokens (no predecessor to validate
// against). Non-first tokens are validated via is_valid_token_pair.
//
// Complexity: O(merge_tree_depth) per call, bounded by max_token_length.
bool iree_tokenizer_bpe_is_first_token_reachable(
    const iree_tokenizer_bpe_model_t* model, uint32_t token) {
  // Use the precomputed token_reachable bitmap. This bitmap was built during
  // model compilation using fixed-point iteration through ALL merges, so it
  // correctly handles tokens that have multiple decomposition paths (marking
  // them reachable if ANY decomposition is reachable).
  return iree_any_bit_set(model->backtrack_tables.token_reachable[token / 64],
                          1ull << (token % 64));
}

// Checks if a specific decomposition (left + right) of a token is reachable.
// This is separate from is_first_token_reachable to allow checking alternative
// decompositions when a token has multiple merges producing it.
bool iree_tokenizer_bpe_is_decomposition_reachable(
    const iree_tokenizer_bpe_model_t* model, uint32_t left, uint32_t right,
    uint32_t token) {
  const uint32_t* effective_rank = model->backtrack_tables.effective_rank;

  // Check for blocking merge at the split boundary.
  //
  // A blocking merge at the L|R boundary can only fire if BOTH boundary base
  // tokens are still available as separate tokens. Each boundary token becomes
  // unavailable when it's first consumed by an internal merge within its
  // respective subtree.
  //
  // Example: "hello" = merge(hel, lo). At the hel|lo boundary, the base tokens
  // are 'l' (from hel = merge(he, l)) and 'l' (from lo = merge(l, o)).
  // The left 'l' is consumed at hel's rank (9). The right 'l' is consumed at
  // lo's rank (2). Merge "l l" at rank 1 fires before either 'l' is consumed,
  // so it blocks.
  //
  // Counter-example: "Ġfibonacci" = merge(Ġfib, onacci). Boundary tokens are
  // 'b' (rightmost of Ġfib) and 'o' (leftmost of onacci). Merge "b o" exists
  // at rank 3282. But 'o' is consumed at rank 6 (by "on" merge within onacci),
  // and 'b' is consumed at rank 427 (by "ib" merge within Ġfib). Since
  // 3282 > min(6, 427) = 6, the "b o" merge cannot fire (o is already consumed
  // by the time rank 3282 is evaluated).
  uint32_t left_boundary = iree_tokenizer_bpe_rightmost_base_token(model, left);
  uint32_t right_boundary =
      iree_tokenizer_bpe_leftmost_base_token(model, right);

  iree_tokenizer_merge_hash_result_t boundary_merge =
      iree_tokenizer_vocab_merge_hash_lookup(
          model->merge_hash, (int32_t)left_boundary, (int32_t)right_boundary);
  if (boundary_merge.result_id >= 0 &&
      (uint32_t)boundary_merge.result_id != token) {
    // There's a different merge at this boundary. It blocks only if it fires
    // before EITHER boundary token is consumed by internal merges.
    uint32_t boundary_rank = effective_rank[(uint32_t)boundary_merge.result_id];
    uint32_t left_boundary_consumed =
        iree_tokenizer_bpe_right_boundary_consumed_rank(model, left);
    uint32_t right_boundary_consumed =
        iree_tokenizer_bpe_left_boundary_consumed_rank(model, right);
    uint32_t min_boundary_consumed =
        left_boundary_consumed < right_boundary_consumed
            ? left_boundary_consumed
            : right_boundary_consumed;
    if (boundary_rank < min_boundary_consumed) {
      return false;  // Blocking merge fires before either boundary is consumed.
    }
  }

  // Check left and right subtrees using precomputed reachability. The bitmap
  // was built considering all possible decompositions, so a token marked
  // reachable has at least one valid BPE path to form it.
  bool left_reachable =
      iree_tokenizer_bpe_is_first_token_reachable(model, left);
  bool right_reachable =
      iree_tokenizer_bpe_is_first_token_reachable(model, right);
  return left_reachable && right_reachable;
}

//===----------------------------------------------------------------------===//
// BPE Backtracking: Table Construction
//===----------------------------------------------------------------------===//

// Builds the split_table (inverse merge table) and effective_rank table.
//
// For each merge in rank order, concatenates left+right token text to find
// the result token ID. The split_table records which two tokens were merged
// to produce each result. The effective_rank records the merge creation order.
//
// Non-participant tokens (added tokens, unused) get split_table = (self, self)
// and effective_rank = 0. Merged tokens get effective_rank = merge_rank + 1
// (1-indexed). Multi-byte base tokens that participate in merges as left/right
// components (but are never produced by a merge) get effective_rank = 1 to
// mark them as matchable by the backtracking algorithm.
iree_status_t iree_tokenizer_bpe_build_backtrack_tables(
    iree_tokenizer_bpe_model_t* model) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_tokenizer_vocab_t* vocab = model->vocab;
  iree_host_size_t vocab_capacity = model->vocab_capacity;
  iree_host_size_t merge_count = iree_tokenizer_vocab_merge_count(vocab);

  // Allocate all backtrack tables in a single slab.
  iree_host_size_t reachable_words = (vocab_capacity + 63) / 64;
  iree_host_size_t slab_size = 0;
  iree_host_size_t split_table_offset = 0;
  iree_host_size_t next_prefix_offset = 0;
  iree_host_size_t effective_rank_offset = 0;
  iree_host_size_t token_reachable_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          /*base_size=*/0, &slab_size,
          IREE_STRUCT_FIELD(vocab_capacity, iree_tokenizer_bpe_split_entry_t,
                            &split_table_offset),
          IREE_STRUCT_FIELD(vocab_capacity, uint32_t, &next_prefix_offset),
          IREE_STRUCT_FIELD(vocab_capacity, uint32_t, &effective_rank_offset),
          IREE_STRUCT_FIELD(reachable_words, uint64_t,
                            &token_reachable_offset)));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(model->allocator, slab_size,
                                &model->backtrack_tables.slab));

  uint8_t* slab = (uint8_t*)model->backtrack_tables.slab;
  model->backtrack_tables.split_table =
      (iree_tokenizer_bpe_split_entry_t*)(slab + split_table_offset);
  model->backtrack_tables.next_prefix_match =
      (uint32_t*)(slab + next_prefix_offset);
  model->backtrack_tables.effective_rank =
      (uint32_t*)(slab + effective_rank_offset);
  model->backtrack_tables.token_reachable =
      (uint64_t*)(slab + token_reachable_offset);

  // Initialize: base tokens map to (self, self), rank 0, no prefix.
  for (iree_host_size_t i = 0; i < vocab_capacity; ++i) {
    model->backtrack_tables.split_table[i].left_id = (uint32_t)i;
    model->backtrack_tables.split_table[i].right_id = (uint32_t)i;
    model->backtrack_tables.effective_rank[i] = 0;
    model->backtrack_tables.next_prefix_match[i] = UINT32_MAX;
  }

  // Walk merges in rank order to populate split_table and effective_rank.
  // Only the first merge producing a given result_id wins (it's the canonical
  // decomposition used during pair validation).
  for (iree_host_size_t rank = 0; rank < merge_count; ++rank) {
    iree_tokenizer_merge_t merge = iree_tokenizer_vocab_merge(vocab, rank);

    // Get result_id directly from merge_hash instead of concatenating strings.
    iree_tokenizer_merge_hash_result_t merge_result =
        iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, merge.left_id,
                                               merge.right_id);
    int32_t result_id = merge_result.result_id;
    if (result_id < 0 || (iree_host_size_t)result_id >= vocab_capacity) {
      continue;
    }

    // The trie may store a different token_id for the same text when duplicate
    // tokens exist (the vocab hash returns the first ID, the trie the last).
    // We must set effective_rank on both so that is_valid_token_pair (which
    // uses the merge_hash's result_id) and backtrack_longest_match (which uses
    // the trie's token_id) both see the correct rank.
    //
    // Walk the trie with the result token's text (already stored in vocab)
    // instead of concatenating left+right into a temporary buffer.
    int32_t trie_result_id = -1;
    {
      iree_string_view_t result_text =
          iree_tokenizer_vocab_token_text(vocab, result_id);
      iree_tokenizer_trie_cursor_t trie_cursor;
      iree_tokenizer_trie_cursor_reset(&trie_cursor, model->trie);
      bool trie_walk_ok = true;
      for (iree_host_size_t ci = 0; ci < result_text.size; ++ci) {
        if (!iree_tokenizer_trie_cursor_advance(
                &trie_cursor, (uint8_t)result_text.data[ci])) {
          trie_walk_ok = false;
          break;
        }
      }
      if (trie_walk_ok) {
        trie_result_id = iree_tokenizer_trie_cursor_token_id(&trie_cursor);
      }
    }

    // First merge producing this result wins (set on vocab_lookup ID).
    if (model->backtrack_tables.effective_rank[result_id] == 0) {
      model->backtrack_tables.split_table[result_id].left_id = merge.left_id;
      model->backtrack_tables.split_table[result_id].right_id = merge.right_id;
      model->backtrack_tables.effective_rank[result_id] = (uint32_t)(rank + 1);
    }

    // Also set on trie ID if it differs (handles duplicate tokens).
    if (trie_result_id >= 0 &&
        (iree_host_size_t)trie_result_id < vocab_capacity &&
        trie_result_id != result_id &&
        model->backtrack_tables.effective_rank[trie_result_id] == 0) {
      model->backtrack_tables.split_table[trie_result_id].left_id =
          merge.left_id;
      model->backtrack_tables.split_table[trie_result_id].right_id =
          merge.right_id;
      model->backtrack_tables.effective_rank[trie_result_id] =
          (uint32_t)(rank + 1);
    }
  }

  // Mark multi-byte merge participants as matchable.
  //
  // Some base vocabulary tokens (e.g., SentencePiece's ▁ character, U+2581)
  // span multiple bytes but are never produced by a merge — they are
  // pre-defined in the vocabulary. However, they DO participate as left/right
  // components in other merges (e.g., "▁" + "Hello" → "▁Hello").
  //
  // The backtracking longest_match only accepts multi-byte trie matches when
  // effective_rank > 0. Without this pass, these base tokens would be rejected
  // by the trie and decomposed into byte-fallback tokens, which is incorrect.
  //
  // Setting effective_rank = 1 is safe for is_valid_token_pair because that
  // function detects base tokens via the split_table self-reference check
  // (split_table[id] == {id, id}), not via effective_rank.
  for (iree_host_size_t rank = 0; rank < merge_count; ++rank) {
    iree_tokenizer_merge_t merge = iree_tokenizer_vocab_merge(vocab, rank);
    if ((iree_host_size_t)merge.left_id < vocab_capacity &&
        model->backtrack_tables.effective_rank[(uint32_t)merge.left_id] == 0) {
      iree_string_view_t text =
          iree_tokenizer_vocab_token_text(vocab, merge.left_id);
      if (text.size > 1) {
        model->backtrack_tables.effective_rank[(uint32_t)merge.left_id] = 1;
      }
    }
    if ((iree_host_size_t)merge.right_id < vocab_capacity &&
        model->backtrack_tables.effective_rank[(uint32_t)merge.right_id] == 0) {
      iree_string_view_t text =
          iree_tokenizer_vocab_token_text(vocab, merge.right_id);
      if (text.size > 1) {
        model->backtrack_tables.effective_rank[(uint32_t)merge.right_id] = 1;
      }
    }
  }

  // Mark multi-byte vocabulary tokens as matchable when their bytes cannot be
  // tokenized via byte_to_token (i.e., no single-byte tokens exist for them).
  //
  // Tokens like CJK characters ("你", "好", etc.) exist in vocabularies like
  // TinyLlama as single tokens but never participate in any merge. Their UTF-8
  // bytes (e.g., 0xE4, 0xBD, 0xA0 for "你") don't have single-byte tokens in
  // the vocabulary. Without this pass, they would have effective_rank == 0 and
  // be rejected by the trie match, falling back to byte-level tokens (<0xNN>)
  // instead of matching directly.
  //
  // Tokens whose bytes DO have single-byte tokens (e.g., "aa" where both bytes
  // have token "a") are NOT marked. These tokens require a merge rule to be
  // produced; otherwise, their bytes are tokenized individually.
  //
  // We exclude tokens with the SPECIAL attribute (BOS, EOS, etc.) since those
  // are typically added tokens that should not be matched during normal
  // tokenization.
  for (iree_host_size_t token_id = 0; token_id < vocab_capacity; ++token_id) {
    if (model->backtrack_tables.effective_rank[token_id] > 0) {
      continue;  // Already marked.
    }
    iree_tokenizer_token_attr_t attrs =
        iree_tokenizer_vocab_token_attrs(vocab, (int32_t)token_id);
    if (attrs & IREE_TOKENIZER_TOKEN_ATTR_SPECIAL) continue;  // Skip special.
    iree_string_view_t text =
        iree_tokenizer_vocab_token_text(vocab, (int32_t)token_id);
    if (text.size <= 1) continue;  // Single-byte tokens always matchable.

    // Check if any byte of this token lacks a single-byte token.
    // If so, this token must be directly matchable (no BPE decomposition).
    bool has_unmatchable_byte = false;
    for (iree_host_size_t i = 0; i < text.size; ++i) {
      uint8_t byte = (uint8_t)text.data[i];
      if (model->byte_to_token[byte] < 0) {
        has_unmatchable_byte = true;
        break;
      }
    }
    if (has_unmatchable_byte) {
      model->backtrack_tables.effective_rank[token_id] = 1;
    }
  }

  // Build next_prefix_match: for each token, walk the trie to find the longest
  // proper prefix that is itself a valid token.
  for (iree_host_size_t token_id = 0; token_id < vocab_capacity; ++token_id) {
    iree_string_view_t text =
        iree_tokenizer_vocab_token_text(vocab, (int32_t)token_id);
    if (text.size <= 1) continue;  // Single-byte tokens have no proper prefix.

    iree_tokenizer_trie_cursor_t cursor;
    iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

    int32_t last_prefix_id = -1;
    for (iree_host_size_t byte_index = 0; byte_index < text.size;
         ++byte_index) {
      if (!iree_tokenizer_trie_cursor_advance(&cursor,
                                              (uint8_t)text.data[byte_index])) {
        break;
      }
      // Any token found before the last byte is a proper prefix.
      if (byte_index < text.size - 1) {
        int32_t prefix_id = iree_tokenizer_trie_cursor_token_id(&cursor);
        if (prefix_id >= 0) {
          last_prefix_id = prefix_id;
        }
      }
    }

    if (last_prefix_id >= 0) {
      model->backtrack_tables.next_prefix_match[token_id] =
          (uint32_t)last_prefix_id;
    }
  }

  // Precompute token reachability to avoid recursive calls during encode.
  // A token is "reachable" if it can appear as the first token in a valid BPE
  // encoding (no blocking merges prevent its formation).
  //
  // We use a fixed-point iteration approach:
  // 1. Initialize: single-codepoint base tokens are always reachable
  // 2. Iterate: for each merge where left and right are both reachable AND
  //    the boundary is not blocked, mark the result as reachable
  // 3. Repeat until no new tokens become reachable
  //
  // This handles tokens with multiple merge paths correctly: a token is
  // reachable if ANY of its decomposition paths has both components reachable
  // and no blocking boundary merge.
  memset(model->backtrack_tables.token_reachable, 0,
         reachable_words * sizeof(uint64_t));

  // Initialize: mark single-codepoint base tokens as reachable.
  for (iree_host_size_t token_id = 0; token_id < vocab_capacity; ++token_id) {
    // Base tokens have split_table[id] == {id, id}.
    if (model->backtrack_tables.split_table[token_id].left_id == token_id) {
      iree_string_view_t token_text =
          iree_tokenizer_vocab_token_text(vocab, (int32_t)token_id);
      // Count codepoints in UTF-8 text.
      iree_host_size_t codepoint_count = 0;
      for (iree_host_size_t i = 0; i < token_text.size; ++i) {
        uint8_t byte = (uint8_t)token_text.data[i];
        if ((byte & 0xC0) != 0x80) {
          codepoint_count++;
        }
      }
      if (codepoint_count <= 1) {
        model->backtrack_tables.token_reachable[token_id / 64] |=
            1ull << (token_id % 64);
      }
    }
  }

  // Iterate until fixed point: propagate reachability through merges.
  bool changed = true;
  while (changed) {
    changed = false;
    for (iree_host_size_t rank = 0; rank < merge_count; ++rank) {
      iree_tokenizer_merge_t merge = iree_tokenizer_vocab_merge(vocab, rank);

      // Get result_id directly from merge_hash instead of concatenating
      // strings.
      iree_tokenizer_merge_hash_result_t merge_result =
          iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash,
                                                 merge.left_id, merge.right_id);
      int32_t result_id = merge_result.result_id;
      if (result_id < 0 || (iree_host_size_t)result_id >= vocab_capacity) {
        continue;
      }

      // Skip if this token is already marked as reachable.
      if (iree_any_bit_set(
              model->backtrack_tables.token_reachable[(uint32_t)result_id / 64],
              1ull << ((uint32_t)result_id % 64))) {
        continue;
      }

      // Check if both left and right components are reachable.
      bool left_reachable = iree_any_bit_set(
          model->backtrack_tables.token_reachable[(uint32_t)merge.left_id / 64],
          1ull << ((uint32_t)merge.left_id % 64));
      bool right_reachable =
          iree_any_bit_set(model->backtrack_tables
                               .token_reachable[(uint32_t)merge.right_id / 64],
                           1ull << ((uint32_t)merge.right_id % 64));
      if (!left_reachable || !right_reachable) {
        continue;
      }

      // Check for blocking merge at the boundary.
      uint32_t left_boundary =
          iree_tokenizer_bpe_rightmost_base_token(model, merge.left_id);
      uint32_t right_boundary =
          iree_tokenizer_bpe_leftmost_base_token(model, merge.right_id);

      iree_tokenizer_merge_hash_result_t boundary_merge =
          iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash,
                                                 (int32_t)left_boundary,
                                                 (int32_t)right_boundary);
      if (boundary_merge.result_id >= 0 &&
          (uint32_t)boundary_merge.result_id != (uint32_t)result_id) {
        // There's a different merge at this boundary. It blocks only if it
        // fires before EITHER boundary token is consumed by internal merges.
        uint32_t boundary_rank =
            model->backtrack_tables
                .effective_rank[(uint32_t)boundary_merge.result_id];
        uint32_t left_boundary_consumed =
            iree_tokenizer_bpe_right_boundary_consumed_rank(model,
                                                            merge.left_id);
        uint32_t right_boundary_consumed =
            iree_tokenizer_bpe_left_boundary_consumed_rank(model,
                                                           merge.right_id);
        uint32_t min_boundary_consumed =
            left_boundary_consumed < right_boundary_consumed
                ? left_boundary_consumed
                : right_boundary_consumed;
        if (boundary_rank < min_boundary_consumed) {
          continue;  // Blocking merge fires first.
        }
      }

      // This decomposition is reachable!
      model->backtrack_tables.token_reachable[(uint32_t)result_id / 64] |=
          1ull << ((uint32_t)result_id % 64);
      changed = true;

      // Update split_table to record this reachable decomposition. The initial
      // split_table records the first (lowest-rank) merge for each token, but
      // that merge path may be unreachable due to blocking boundary merges. For
      // example, "cali" has first merge (ca, li) blocked by a+l→al at the
      // boundary, but is reachable via (c, ali) at a higher rank. Storing the
      // first REACHABLE merge ensures that suffix_blocked correctly walks the
      // token's actual BPE merge tree when collecting suffixes.
      //
      // This is safe within the fixed-point loop because:
      // - We only update when a token first becomes reachable (never again).
      // - The rightmost/leftmost base tokens are invariant to decomposition
      //   (they depend on the character sequence, not the merge tree).
      // - The boundary consumed ranks become more accurate, reflecting the
      //   actual BPE merge path rather than an unreachable one.
      model->backtrack_tables.split_table[(uint32_t)result_id].left_id =
          (uint32_t)merge.left_id;
      model->backtrack_tables.split_table[(uint32_t)result_id].right_id =
          (uint32_t)merge.right_id;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
