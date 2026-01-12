// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/bpe.h"

//===----------------------------------------------------------------------===//
// Merge Lookup Hash Table
//===----------------------------------------------------------------------===//

// Hash table entry for merge lookup.
typedef struct iree_bpe_merge_entry_t {
  uint32_t left_id;
  uint32_t right_id;
  int32_t rank;  // -1 if slot is empty.
} iree_bpe_merge_entry_t;

// Hash function for merge pair.
static uint32_t iree_bpe_merge_hash(uint32_t left_id, uint32_t right_id) {
  // FNV-1a style mixing.
  uint64_t key = ((uint64_t)left_id << 32) | right_id;
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdULL;
  key ^= key >> 33;
  key *= 0xc4ceb9fe1a85ec53ULL;
  key ^= key >> 33;
  return (uint32_t)key;
}

//===----------------------------------------------------------------------===//
// BPE Symbol (linked list node)
//===----------------------------------------------------------------------===//

// Symbol in the merge buffer (doubly-linked list).
typedef struct iree_bpe_symbol_t {
  int32_t token_id;
  struct iree_bpe_symbol_t* prev;
  struct iree_bpe_symbol_t* next;
} iree_bpe_symbol_t;

//===----------------------------------------------------------------------===//
// BPE Encoder State
//===----------------------------------------------------------------------===//

struct iree_tokenizer_bpe_state_t {
  iree_allocator_t allocator;
  const iree_tokenizer_vocab_t* vocab;

  // Merge lookup hash table.
  iree_bpe_merge_entry_t* merge_table;
  iree_host_size_t merge_table_size;  // Must be power of 2.

  // End-of-word suffix (e.g., "</w>" for CLIP).
  // Empty if not used. Appended to each word before tokenization.
  char end_of_word_suffix[16];
  iree_host_size_t end_of_word_suffix_length;
};

// Looks up the rank of a merge pair.
// Returns -1 if no merge exists for the pair.
static int32_t iree_bpe_lookup_merge(const iree_tokenizer_bpe_state_t* state,
                                     uint32_t left_id, uint32_t right_id) {
  if (!state->merge_table || state->merge_table_size == 0) {
    return -1;
  }

  uint32_t mask = (uint32_t)(state->merge_table_size - 1);
  uint32_t hash = iree_bpe_merge_hash(left_id, right_id);
  uint32_t idx = hash & mask;

  for (iree_host_size_t i = 0; i < state->merge_table_size; ++i) {
    const iree_bpe_merge_entry_t* entry = &state->merge_table[idx];
    if (entry->rank < 0) {
      return -1;  // Empty slot, not found.
    }
    if (entry->left_id == left_id && entry->right_id == right_id) {
      return entry->rank;
    }
    idx = (idx + 1) & mask;  // Linear probing.
  }
  return -1;
}

// Inserts a merge into the hash table.
static void iree_bpe_insert_merge(iree_tokenizer_bpe_state_t* state,
                                  uint32_t left_id, uint32_t right_id,
                                  int32_t rank) {
  uint32_t mask = (uint32_t)(state->merge_table_size - 1);
  uint32_t hash = iree_bpe_merge_hash(left_id, right_id);
  uint32_t idx = hash & mask;

  while (state->merge_table[idx].rank >= 0) {
    idx = (idx + 1) & mask;
  }

  state->merge_table[idx].left_id = left_id;
  state->merge_table[idx].right_id = right_id;
  state->merge_table[idx].rank = rank;
}

//===----------------------------------------------------------------------===//
// BPE State Creation
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_bpe_state_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_allocator_t allocator,
    iree_tokenizer_bpe_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = NULL;

  // Guard against overflow in merge_count * 2 (practically impossible but
  // buffer overflows are unacceptable).
  iree_host_size_t merge_count = iree_tokenizer_vocab_merge_count(vocab);
  if (merge_count > IREE_HOST_SIZE_MAX / 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "merge count overflow (%" PRIhsz " merges)",
                            merge_count);
  }

  // Allocate state.
  iree_tokenizer_bpe_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->allocator = allocator;
  state->vocab = vocab;

  // Build merge lookup table.
  if (merge_count > 0) {
    // Size table to ~50% load factor, round up to power of 2.
    iree_host_size_t table_size = 1;
    while (table_size < merge_count * 2) {
      table_size *= 2;
    }

    iree_status_t status = iree_allocator_malloc(
        allocator, table_size * sizeof(iree_bpe_merge_entry_t),
        (void**)&state->merge_table);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, state);
      return status;
    }

    state->merge_table_size = table_size;

    // Initialize all slots as empty.
    for (iree_host_size_t i = 0; i < table_size; ++i) {
      state->merge_table[i].rank = -1;
    }

    // Insert all merges.
    for (iree_host_size_t rank = 0; rank < merge_count; ++rank) {
      iree_tokenizer_merge_t merge = iree_tokenizer_vocab_merge(vocab, rank);
      iree_bpe_insert_merge(state, merge.left_id, merge.right_id,
                            (int32_t)rank);
    }
  }

  *out_state = state;
  return iree_ok_status();
}

void iree_tokenizer_bpe_state_free(iree_tokenizer_bpe_state_t* state) {
  if (!state) return;
  iree_allocator_t allocator = state->allocator;

  if (state->merge_table) {
    iree_allocator_free(allocator, state->merge_table);
  }
  iree_allocator_free(allocator, state);
}

iree_status_t iree_tokenizer_bpe_state_set_end_of_word_suffix(
    iree_tokenizer_bpe_state_t* state, iree_string_view_t suffix) {
  IREE_ASSERT_ARGUMENT(state);
  if (suffix.size > sizeof(state->end_of_word_suffix)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "end_of_word_suffix too long (%" PRIhsz
                            " bytes, max %zu)",
                            suffix.size, sizeof(state->end_of_word_suffix));
  }
  memcpy(state->end_of_word_suffix, suffix.data, suffix.size);
  state->end_of_word_suffix_length = suffix.size;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BPE Encoding
//===----------------------------------------------------------------------===//

// Maximum symbols (bytes or chars) in a single word during BPE encoding.
// Risk: MODERATE for multilingual text without proper vocab coverage.
// Acceptable because:
// - English/common languages: Safe (words rarely exceed 512 symbols)
// - Byte-level models: Safe (designed for this)
// - Multi-byte fallback only hits limit with 256+ unknown bytes in one word
// - Error is clearly reported with RESOURCE_EXHAUSTED
#define IREE_TOKENIZER_BPE_MAX_SYMBOLS 512

// Maximum size of a merged token during BPE encoding.
// Risk: LOW - Real-world vocabs max ~100-150 byte merges.
// Acceptable because:
// - GPT-2: max ~60-70 bytes per token
// - Llama 2: max ~100 bytes per token
// - BERT: max ~40 bytes per token
// - Would only overflow with adversarial vocab (150+150 byte tokens merging)
// - Error is clearly reported with RESOURCE_EXHAUSTED
#define IREE_TOKENIZER_BPE_MAX_MERGED_TOKEN_SIZE 256

// Maximum word size including end_of_word_suffix.
#define IREE_TOKENIZER_BPE_MAX_WORD_SIZE 8192

// Maximum merge iterations before aborting.
// Risk: LOW - defensive limit against malicious cyclic merge rules.
// Acceptable because:
// - With N symbols, at most N-1 merges can occur (each merge reduces count by
// 1)
// - IREE_TOKENIZER_BPE_MAX_SYMBOLS is 512, so max legitimate merges = 511
// - 10000 is generous headroom while detecting infinite loops quickly
// - Error is clearly reported with RESOURCE_EXHAUSTED
#define IREE_TOKENIZER_BPE_MAX_MERGE_ITERATIONS 10000

iree_status_t iree_tokenizer_bpe_encode_word(
    const iree_tokenizer_bpe_state_t* state, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_ids);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;

  if (word.size == 0) {
    return iree_ok_status();
  }

  // Append end_of_word_suffix if configured (e.g., "</w>" for CLIP).
  char word_buffer[IREE_TOKENIZER_BPE_MAX_WORD_SIZE];
  iree_string_view_t effective_word = word;
  if (state->end_of_word_suffix_length > 0) {
    iree_host_size_t total_size = word.size + state->end_of_word_suffix_length;
    if (total_size > sizeof(word_buffer)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "word + suffix exceeds maximum size (%" PRIhsz
                              " bytes, max %zu)",
                              total_size, sizeof(word_buffer));
    }
    memcpy(word_buffer, word.data, word.size);
    memcpy(word_buffer + word.size, state->end_of_word_suffix,
           state->end_of_word_suffix_length);
    effective_word = iree_make_string_view(word_buffer, total_size);
  }

  const iree_tokenizer_vocab_t* vocab = state->vocab;

  // Fast path: Try looking up the entire word as a single token.
  // This handles CLIP-style vocabularies where "a</w>" is a single token.
  int32_t whole_word_id = iree_tokenizer_vocab_lookup(vocab, effective_word);
  if (whole_word_id >= 0) {
    if (max_ids < 1) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small for single token");
    }
    out_ids[0] = whole_word_id;
    *out_count = 1;
    return iree_ok_status();
  }

  // Allocate symbol array on stack.
  iree_bpe_symbol_t symbols[IREE_TOKENIZER_BPE_MAX_SYMBOLS];
  iree_host_size_t symbol_count = 0;

  // Phase 1: Initialize symbols from characters.
  // Try to find each character as a token. If not found, use byte fallback.
  iree_host_size_t position = 0;
  while (position < effective_word.size &&
         symbol_count < IREE_TOKENIZER_BPE_MAX_SYMBOLS) {
    // Determine UTF-8 character length.
    uint8_t byte = (uint8_t)effective_word.data[position];
    iree_host_size_t char_length = 1;
    if ((byte & 0xE0) == 0xC0)
      char_length = 2;
    else if ((byte & 0xF0) == 0xE0)
      char_length = 3;
    else if ((byte & 0xF8) == 0xF0)
      char_length = 4;

    if (position + char_length > effective_word.size) {
      char_length = effective_word.size - position;  // Truncate invalid UTF-8.
    }

    // Try to look up the character as a token.
    iree_string_view_t char_sv =
        iree_make_string_view(effective_word.data + position, char_length);
    int32_t token_id = iree_tokenizer_vocab_lookup(vocab, char_sv);

    if (token_id >= 0) {
      symbols[symbol_count].token_id = token_id;
      symbols[symbol_count].prev =
          (symbol_count > 0) ? &symbols[symbol_count - 1] : NULL;
      symbols[symbol_count].next = NULL;
      if (symbol_count > 0) {
        symbols[symbol_count - 1].next = &symbols[symbol_count];
      }
      ++symbol_count;
    } else {
      // Byte fallback: encode each byte as <0xNN>.
      for (iree_host_size_t i = 0;
           i < char_length && symbol_count < IREE_TOKENIZER_BPE_MAX_SYMBOLS;
           ++i) {
        char byte_str[8];
        snprintf(byte_str, sizeof(byte_str), "<0x%02X>",
                 (unsigned char)effective_word.data[position + i]);
        token_id = iree_tokenizer_vocab_lookup(
            vocab, iree_make_cstring_view(byte_str));

        if (token_id < 0) {
          // No byte fallback token, return unknown.
          iree_tokenizer_special_ids_t special_ids =
              iree_tokenizer_vocab_special_ids(vocab);
          if (special_ids.unk < 0) {
            return iree_make_status(
                IREE_STATUS_NOT_FOUND,
                "cannot encode byte and no [UNK] or byte fallback available");
          }
          token_id = special_ids.unk;
        }

        symbols[symbol_count].token_id = token_id;
        symbols[symbol_count].prev =
            (symbol_count > 0) ? &symbols[symbol_count - 1] : NULL;
        symbols[symbol_count].next = NULL;
        if (symbol_count > 0) {
          symbols[symbol_count - 1].next = &symbols[symbol_count];
        }
        ++symbol_count;
      }
    }
    position += char_length;
  }

  // Check for truncation (input remaining but symbol array full).
  if (position < effective_word.size &&
      symbol_count >= IREE_TOKENIZER_BPE_MAX_SYMBOLS) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "word exceeds maximum symbol count (%d); input "
                            "truncated at byte %" PRIhsz,
                            IREE_TOKENIZER_BPE_MAX_SYMBOLS, position);
  }

  if (symbol_count == 0) {
    return iree_ok_status();
  }

  // Phase 2: Iteratively apply merges.
  // Find the best merge (lowest rank), apply it, repeat.
  bool changed = true;
  iree_host_size_t iterations = 0;
  while (changed) {
    if (++iterations > IREE_TOKENIZER_BPE_MAX_MERGE_ITERATIONS) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "BPE merge iteration limit exceeded (%" PRIhsz
          ") - possible cyclic merge rules in vocabulary",
          (iree_host_size_t)IREE_TOKENIZER_BPE_MAX_MERGE_ITERATIONS);
    }
    changed = false;
    int32_t best_rank = INT32_MAX;
    iree_bpe_symbol_t* best_left = NULL;

    // Find the best merge across all adjacent pairs.
    for (iree_bpe_symbol_t* sym = &symbols[0]; sym && sym->next;
         sym = sym->next) {
      if (sym->token_id < 0) continue;  // Deleted symbol.
      iree_bpe_symbol_t* right = sym->next;
      while (right && right->token_id < 0) {
        right = right->next;  // Skip deleted symbols.
      }
      if (!right) continue;

      int32_t rank = iree_bpe_lookup_merge(state, (uint32_t)sym->token_id,
                                           (uint32_t)right->token_id);
      if (rank >= 0 && rank < best_rank) {
        best_rank = rank;
        best_left = sym;
      }
    }

    if (best_left) {
      // Apply the merge: left + right -> merged.
      iree_bpe_symbol_t* left = best_left;
      iree_bpe_symbol_t* right = left->next;
      while (right && right->token_id < 0) {
        right = right->next;
      }

      // Look up the merged token.
      iree_string_view_t left_text =
          iree_tokenizer_vocab_token_text(vocab, left->token_id);
      iree_string_view_t right_text =
          iree_tokenizer_vocab_token_text(vocab, right->token_id);

      // Build merged text.
      char merged_text[IREE_TOKENIZER_BPE_MAX_MERGED_TOKEN_SIZE];
      if (left_text.size + right_text.size >= sizeof(merged_text)) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "BPE merge would exceed maximum token length (%" PRIhsz
            " + %" PRIhsz " bytes)",
            left_text.size, right_text.size);
      }
      memcpy(merged_text, left_text.data, left_text.size);
      memcpy(merged_text + left_text.size, right_text.data, right_text.size);
      iree_string_view_t merged_sv =
          iree_make_string_view(merged_text, left_text.size + right_text.size);

      int32_t merged_id = iree_tokenizer_vocab_lookup(vocab, merged_sv);
      if (merged_id < 0) {
        // Merge rules and vocabulary are inconsistent - this indicates a
        // corrupt tokenizer.json or a bug in the JSON parser.
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "BPE merge produced token not in vocabulary: '%.*s' + '%.*s' = "
            "'%.*s' (token IDs %d + %d)",
            (int)left_text.size, left_text.data, (int)right_text.size,
            right_text.data, (int)merged_sv.size, merged_sv.data,
            left->token_id, right->token_id);
      }

      // Replace left with merged, mark right as deleted.
      left->token_id = merged_id;
      right->token_id = -1;  // Mark as deleted.

      // Update linked list to skip deleted node.
      left->next = right->next;
      if (right->next) {
        right->next->prev = left;
      }

      changed = true;
    }
  }

  // Phase 3: Output the remaining symbols.
  iree_host_size_t output_count = 0;
  for (iree_bpe_symbol_t* sym = &symbols[0]; sym; sym = sym->next) {
    if (sym->token_id < 0) continue;  // Skip deleted.
    if (output_count >= max_ids) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small for BPE tokens");
    }
    out_ids[output_count++] = sym->token_id;
  }

  *out_count = output_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BPE Tokenizer (unified interface)
//===----------------------------------------------------------------------===//

// Extended tokenizer structure for BPE.
typedef struct iree_tokenizer_bpe_t {
  iree_tokenizer_t base;
  iree_tokenizer_bpe_state_t* state;  // Owned.
} iree_tokenizer_bpe_t;

static void iree_tokenizer_bpe_destroy_impl(iree_tokenizer_t* tokenizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_bpe_t* bpe = (iree_tokenizer_bpe_t*)tokenizer;
  // Destroy BPE-specific state. Base handles vocab and struct.
  if (bpe->state) {
    iree_tokenizer_bpe_state_free(bpe->state);
  }
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_bpe_encode_word_impl(
    const iree_tokenizer_t* tokenizer, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count) {
  const iree_tokenizer_bpe_t* bpe = (const iree_tokenizer_bpe_t*)tokenizer;
  return iree_tokenizer_bpe_encode_word(bpe->state, word, out_ids, max_ids,
                                        out_count);
}

static const iree_tokenizer_vtable_t iree_tokenizer_bpe_vtable = {
    .destroy = iree_tokenizer_bpe_destroy_impl,
    .encode_word = iree_tokenizer_bpe_encode_word_impl,
};

iree_status_t iree_tokenizer_bpe_allocate(iree_tokenizer_vocab_t* vocab,
                                          iree_allocator_t allocator,
                                          iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the extended tokenizer struct.
  // Note: BPE with zero merge rules is valid - it simply won't merge any tokens
  // and will output base vocabulary tokens directly. This is useful for testing
  // and for vocabularies that are already in their final merged form.
  iree_tokenizer_bpe_t* tokenizer = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator, sizeof(iree_tokenizer_bpe_t), (void**)&tokenizer);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_free(vocab);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Initialize base (stores vocab - from here, iree_tokenizer_free handles it).
  iree_tokenizer_initialize(&tokenizer->base, &iree_tokenizer_bpe_vtable,
                            allocator, vocab, /*transform=*/NULL,
                            /*decoder=*/NULL, /*postprocessor=*/NULL);

  // Create BPE state.
  status =
      iree_tokenizer_bpe_state_allocate(vocab, allocator, &tokenizer->state);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_free(&tokenizer->base);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_tokenizer = &tokenizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_tokenizer_bpe_set_end_of_word_suffix(
    iree_tokenizer_t* tokenizer, iree_string_view_t suffix) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  iree_tokenizer_bpe_t* bpe = (iree_tokenizer_bpe_t*)tokenizer;
  return iree_tokenizer_bpe_state_set_end_of_word_suffix(bpe->state, suffix);
}
