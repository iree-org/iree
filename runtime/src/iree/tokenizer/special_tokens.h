// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Special token matching for pre-normalization literal token recognition.
//
// Special tokens (like <|endoftext|>, [CLS], <bos>) must be matched in raw
// input BEFORE normalization runs, otherwise pre-tokenizers like ByteLevel
// would split them (e.g., treating | as a word boundary).
//
// This component provides:
// - Two-level prefix index for O(1) rejection of non-matching bytes
// - B-string layout for cache-efficient token storage
// - Longest-match semantics with streaming support
//
// Design principles:
// - Hot path (99.9%): 1 cache line, 1 dependent fetch (first byte not in any
//   bucket)
// - Warm path (0.09%): 2 cache lines (first byte matches but prefix doesn't,
//   e.g., `<` in code/XML)
// - Cold path (0.01%): linear scan of bucket (actual special token match)
//
// Real-world data (from HuggingFace tokenizers):
// - DeepSeek-V3: 804 special tokens, all start with `<｜` (fullwidth pipe)
// - Llama-3.1: 256 special tokens, all start with `<|`
// - BERT: 5 special tokens, all start with `[`
//
// The two-level prefix index means:
// - For code/XML with many `<`: second byte check rejects immediately
// - DeepSeek's 804 tokens only scan when input has `<｜` (never in normal text)

#ifndef IREE_TOKENIZER_SPECIAL_TOKENS_H_
#define IREE_TOKENIZER_SPECIAL_TOKENS_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Special Token Flags
//===----------------------------------------------------------------------===//

// Flags controlling special token matching behavior.
enum iree_tokenizer_special_token_flag_bits_e {
  IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE = 0u,
  // Match only if preceded by whitespace or start of input.
  IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP = 1u << 0,
  // Match only if followed by whitespace or end of input.
  IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP = 1u << 1,
  // Match only whole words (both lstrip AND rstrip semantics + word boundary).
  IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_SINGLE_WORD = 1u << 2,
};
typedef uint8_t iree_tokenizer_special_token_flags_t;

//===----------------------------------------------------------------------===//
// Special Tokens Type
//===----------------------------------------------------------------------===//

// Maximum number of distinct prefix buckets. Real-world tokenizers have at
// most 2-3 distinct first bytes for special tokens.
#define IREE_TOKENIZER_SPECIAL_TOKENS_MAX_BUCKETS 16

// Sentinel value indicating no bucket exists for a first byte.
#define IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET 0xFF

// A bucket groups tokens sharing a common prefix for efficient matching.
// Tokens within a bucket are sorted by length (descending) so the first match
// found is the longest match.
typedef struct iree_tokenizer_special_tokens_bucket_t {
  // Common prefix bytes (up to 4).
  uint8_t prefix[4];
  // Number of prefix bytes to check (1-4).
  uint8_t prefix_length;
  // First token index in this bucket.
  uint16_t start;
  // One past last token index.
  uint16_t end;
} iree_tokenizer_special_tokens_bucket_t;

// Special tokens collection with two-level prefix index.
//
// Token strings are stored in B-string layout: [length:uint8][content:bytes].
// No NUL terminators, enabling direct memcmp. Tokens are sorted by prefix
// bucket, then by length (descending) within each bucket.
//
// The two-level index provides O(1) rejection:
//   Level 1: first_byte_to_bucket[byte] -> bucket index (0xFF = none)
//   Level 2: bucket.prefix -> multi-byte prefix check before scanning
typedef struct iree_tokenizer_special_tokens_t {
  iree_host_size_t count;       // Total number of special tokens.
  iree_host_size_t max_length;  // Longest token (for buffer sizing).
  iree_host_size_t min_length;  // Shortest token (for early rejection).

  // Level 1 index: first byte -> bucket index (0xFF = no bucket).
  uint8_t first_byte_to_bucket[256];

  // Level 2 index: bucket prefix and token range.
  iree_tokenizer_special_tokens_bucket_t
      buckets[IREE_TOKENIZER_SPECIAL_TOKENS_MAX_BUCKETS];
  uint8_t bucket_count;

  // Single slab allocation containing
  // [ids][flags][bstring_offsets][bstring_data]. NULL if empty (no allocation
  // made).
  void* slab;

  // Pointers into slab (do not free separately).
  // ids[i] = token ID for i-th B-string.
  iree_tokenizer_token_id_t* ids;
  iree_tokenizer_special_token_flags_t* flags;  // flags[i] = matching flags.
  // bstring_offsets[i] = byte offset of token i in data.
  uint32_t* bstring_offsets;
  iree_byte_span_t data;  // B-strings: [len0][str0...][len1][str1...]

  // Allocator used for slab allocation.
  iree_allocator_t allocator;
} iree_tokenizer_special_tokens_t;

// Returns true if the special tokens collection is empty.
static inline bool iree_tokenizer_special_tokens_is_empty(
    const iree_tokenizer_special_tokens_t* special_tokens) {
  return special_tokens->count == 0;
}

// Initializes an empty special tokens collection.
// No allocations are made; the collection can be used immediately but will
// match nothing.
void iree_tokenizer_special_tokens_initialize(
    iree_tokenizer_special_tokens_t* out_special_tokens);

// Deinitializes a special tokens collection, freeing any allocated memory.
void iree_tokenizer_special_tokens_deinitialize(
    iree_tokenizer_special_tokens_t* special_tokens);

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

// Entry for a token being built. References into the shared string table.
typedef struct iree_tokenizer_special_tokens_builder_entry_t {
  iree_tokenizer_token_id_t id;
  uint32_t string_offset;                      // Offset into string_data.
  uint16_t string_length;                      // Length of token content.
  iree_tokenizer_special_token_flags_t flags;  // Matching behavior flags.
} iree_tokenizer_special_tokens_builder_entry_t;

// Builder for constructing a special tokens collection.
// Can be embedded in other structures or stack-allocated.
//
// Usage:
//   1. Initialize builder with _builder_initialize()
//   2. Add tokens with _builder_add()
//   3. Build final collection with _builder_build()
//   4. Deinitialize builder with _builder_deinitialize()
typedef struct iree_tokenizer_special_tokens_builder_t {
  iree_allocator_t allocator;

  // Token entries (offset/length pairs into string_data).
  iree_tokenizer_special_tokens_builder_entry_t* entries;
  iree_host_size_t entry_count;
  iree_host_size_t entry_capacity;

  // Shared string table for all token contents.
  uint8_t* string_data;
  iree_host_size_t string_size;
  iree_host_size_t string_capacity;
} iree_tokenizer_special_tokens_builder_t;

// Initializes a special tokens builder.
// The builder must be deinitialized with _builder_deinitialize() after use.
void iree_tokenizer_special_tokens_builder_initialize(
    iree_allocator_t allocator,
    iree_tokenizer_special_tokens_builder_t* out_builder);

// Adds a special token to the builder.
// |content| is the literal token string (not NUL-terminated).
// |token_id| is the vocabulary ID for this token.
// |flags| controls matching behavior (lstrip/rstrip/single_word).
// Tokens can be added in any order; they will be sorted during build.
iree_status_t iree_tokenizer_special_tokens_builder_add(
    iree_tokenizer_special_tokens_builder_t* builder,
    iree_string_view_t content, iree_tokenizer_token_id_t token_id,
    iree_tokenizer_special_token_flags_t flags);

// Builds the final special tokens collection from all added tokens.
// |allocator| is used for the output collection's internal allocations.
// After this call, the builder can be deinitialized or reset for another build.
// The output collection must be deinitialized with
// _special_tokens_deinitialize().
iree_status_t iree_tokenizer_special_tokens_builder_build(
    iree_tokenizer_special_tokens_builder_t* builder,
    iree_allocator_t allocator,
    iree_tokenizer_special_tokens_t* out_special_tokens);

// Deinitializes the builder, freeing all internal resources.
// The builder struct itself is not freed (caller owns the storage).
void iree_tokenizer_special_tokens_builder_deinitialize(
    iree_tokenizer_special_tokens_builder_t* builder);

//===----------------------------------------------------------------------===//
// Matching
//===----------------------------------------------------------------------===//

// Result of a special token match attempt.
typedef enum iree_tokenizer_special_tokens_match_result_e {
  // No special token matches at position 0. The caller should pass the first
  // byte through to the normalizer and retry matching at the next position.
  IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH = 0,

  // A special token fully matched. The match_length and token_id outputs
  // contain the match details. The caller should emit the token and advance
  // by match_length bytes.
  IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED = 1,

  // Input is a prefix of one or more special tokens but too short to determine
  // if it's a complete match. The caller should buffer the input and provide
  // more data. This only happens at chunk boundaries in streaming mode.
  IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE = 2,
} iree_tokenizer_special_tokens_match_result_t;

// Forward declaration for streaming state (defined below).
struct iree_tokenizer_special_tokens_encode_state_t;

// Attempts to match a special token at the beginning of |input|.
//
// The |state| parameter tracks partial matches across calls. Initialize to zero
// before the first call. When continuing after NEED_MORE, pass the same state.
//
// Returns:
//   NO_MATCH: No token matches. Pass input[0] to normalizer, retry at input+1.
//             If state->match_position > 0, use get_partial() to recover the
//             bytes that were consumed before the mismatch.
//   MATCHED: Token matched. *out_length = bytes consumed from THIS input.
//   NEED_MORE: Partial match. Caller should consume input, provide more bytes.
//
// Streaming contract for NEED_MORE:
//   - Caller consumes ALL of |input| (those bytes are now tracked in state)
//   - Next call provides only new bytes (not the previously consumed ones)
//   - state->match_position tracks total bytes matched so far
//
// This function implements longest-match semantics.
iree_tokenizer_special_tokens_match_result_t
iree_tokenizer_special_tokens_match(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_string_view_t input, iree_host_size_t* out_length,
    iree_tokenizer_token_id_t* out_id,
    struct iree_tokenizer_special_tokens_encode_state_t* state);

// Returns the number of bytes at the start of |input| that cannot begin any
// special token. These bytes can be safely passed to the normalizer without
// checking for special tokens at each position.
//
// This enables efficient batching: instead of checking each byte, scan ahead
// to find the first byte that could start a special token, process all bytes
// before it in one batch.
//
// Returns input.size if no byte could start a special token.
//
// Performance: When bucket_count == 1 (common case: all special tokens share
// the same first byte like '<' for GPT/Llama or '[' for BERT), uses memchr()
// which is SIMD-optimized. For multiple first bytes, falls back to a lookup
// table scan.
static inline iree_host_size_t iree_tokenizer_special_tokens_safe_prefix_length(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_string_view_t input) {
  if (special_tokens->count == 0) return input.size;

  // Fast path: single bucket means all special tokens share the same first
  // byte. Use memchr which is SIMD-optimized (16-32 bytes per instruction).
  // Real-world tokenizers almost always have a single first byte
  // (e.g., '<' for GPT/Llama/DeepSeek, '[' for BERT).
  if (IREE_LIKELY(special_tokens->bucket_count == 1)) {
    uint8_t first_byte = special_tokens->buckets[0].prefix[0];
    const void* found = memchr(input.data, first_byte, input.size);
    return found ? (iree_host_size_t)((const char*)found - input.data)
                 : input.size;
  }

  // General path: multiple possible first bytes. Scan using lookup table.
  // This is O(n) but the table lookup is cache-friendly (256 bytes fits in
  // one cache line on most architectures).
  for (iree_host_size_t i = 0; i < input.size; ++i) {
    if (special_tokens->first_byte_to_bucket[(uint8_t)input.data[i]] !=
        IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET) {
      return i;
    }
  }
  return input.size;
}

//===----------------------------------------------------------------------===//
// Encode State (for streaming)
//===----------------------------------------------------------------------===//

// Per-encode state for special token matching in streaming mode.
// Tracks partial matches that span chunk boundaries.
//
// No separate buffer is needed: when a partial match fails, the matched prefix
// can be reconstructed from the specific token that was being matched.
typedef struct iree_tokenizer_special_tokens_encode_state_t {
  // Number of bytes matched so far in a partial match (0 = no partial).
  uint16_t match_position;

  // Index of the specific token we're partially matching. When NEED_MORE is
  // returned, this identifies which token's content to use for reconstruction
  // if the match ultimately fails. This is NOT just the first token in the
  // bucket - different tokens in a bucket can have different content after
  // the shared prefix.
  uint16_t partial_token_index;

  // Drain position: how many bytes of the partial match have been drained.
  // Used during NO_MATCH recovery to emit bytes in correct order.
  // When drain_position == match_position, draining is complete.
  uint16_t drain_position;

  // Previous byte before current position (for lstrip/single_word checks).
  // 0 = start of input (no previous byte). Stored as value+1 so 0 is unset.
  uint8_t prev_byte_plus_one;

  // True if this is the start of input (before any bytes have been processed).
  bool at_start_of_input;
} iree_tokenizer_special_tokens_encode_state_t;

// Initializes encode state for a new encoding session.
static inline void iree_tokenizer_special_tokens_encode_state_initialize(
    iree_tokenizer_special_tokens_encode_state_t* out_state) {
  out_state->match_position = 0;
  out_state->partial_token_index = 0;
  out_state->drain_position = 0;
  out_state->prev_byte_plus_one = 0;  // No previous byte yet.
  out_state->at_start_of_input = true;
}

// Returns true if there is a partial match in progress.
static inline bool iree_tokenizer_special_tokens_encode_state_has_partial(
    const iree_tokenizer_special_tokens_encode_state_t* state) {
  return state->match_position > 0;
}

// Updates the previous byte context after processing a byte.
// Called by the tokenizer after passing a byte through (not a special token).
static inline void iree_tokenizer_special_tokens_encode_state_update_prev_byte(
    iree_tokenizer_special_tokens_encode_state_t* state, uint8_t byte) {
  state->prev_byte_plus_one = byte + 1;
  state->at_start_of_input = false;
}

// Returns true if the byte is a word boundary character (whitespace or punct).
// Used for lstrip/rstrip/single_word checks.
static inline bool iree_tokenizer_is_word_boundary_byte(uint8_t byte) {
  // ASCII whitespace: space, tab, newline, carriage return, form feed, vtab.
  if (byte <= 0x20) return true;  // Control chars and space.
  // Common ASCII punctuation that serves as word boundaries.
  // Note: This is intentionally simple - HuggingFace uses regex \w which is
  // locale-dependent. We use a conservative ASCII-only definition.
  return (byte >= 0x21 && byte <= 0x2F) ||  // !"#$%&'()*+,-./
         (byte >= 0x3A && byte <= 0x40) ||  // :;<=>?@
         (byte >= 0x5B && byte <= 0x60) ||  // [\]^_`
         (byte >= 0x7B && byte <= 0x7E);    // {|}~
}

// Gets the partial match prefix bytes for reconstruction on match failure.
// Returns the number of bytes written to |out_buffer|.
// |out_buffer| must have at least |state->match_position| bytes available.
// Only valid when has_partial() returns true.
iree_host_size_t iree_tokenizer_special_tokens_encode_state_get_partial(
    const iree_tokenizer_special_tokens_encode_state_t* state,
    const iree_tokenizer_special_tokens_t* special_tokens, uint8_t* out_buffer);

// Clears any partial match state.
static inline void iree_tokenizer_special_tokens_encode_state_clear_partial(
    iree_tokenizer_special_tokens_encode_state_t* state) {
  state->match_position = 0;
  state->drain_position = 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SPECIAL_TOKENS_H_
