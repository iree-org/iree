// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/precompiled.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Maximum bytes buffered for pattern matching across chunk boundaries.
//
// This value comes directly from SentencePiece's kMaxTrieResultsSize constant
// defined in normalizer.h as "Maximum size of the return value of Trie, which
// corresponds to the maximum size of shared common prefix in the chars map."
//
// Source: https://github.com/google/sentencepiece/blob/master/src/normalizer.h
//   static constexpr int kMaxTrieResultsSize = 32;
//
// Empirical verification across 9 production tokenizers (T5, XLM-R, CamemBERT,
// DeBERTa-v3, BGE-M3, multilingual-e5, paraphrase-multilingual-MiniLM):
//   - Maximum observed key depth: 11 bytes
//   - All tokenizers use identical NFKC precompiled charsmap (~60KB pool)
//
// Why 32 bytes is sufficient:
//   - Trie keys represent Unicode normalization mappings (NFKC/NFKD)
//   - Keys are UTF-8 encoded source sequences to normalize
//   - Maximum UTF-8 codepoint: 4 bytes
//   - Longest NFKC mappings: 2-3 composed characters (CJK compatibility,
//     ligatures)
//   - 32 bytes provides 3x margin over empirical maximum of 11 bytes
//
// Streaming guarantee: Any pattern that could match starting at byte N will
// be fully contained in the overlap buffer when processing resumes at the
// next chunk. No pattern can span more than 32 bytes, so matching behavior
// is identical to batch mode.
#define IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP 32

// Maximum bytes buffered for replacement strings awaiting output.
//
// When a trie match produces a replacement string that doesn't fit in the
// caller's output buffer, the replacement is held in the pending buffer
// until subsequent calls drain it.
//
// Format constraint (hard limit):
//   The precompiled charsmap pool uses length-prefixed format where each
//   string's length is stored in a uint8_t. This limits any single
//   replacement string to 255 bytes maximum.
//
// Empirical verification across 9 production tokenizers:
//   - Maximum observed replacement length: 33 bytes
//   - These occur for complex CJK compatibility character decompositions
//   - Example: single compatibility ideograph expanding to multiple characters
//
// Why 64 bytes is sufficient:
//   - Empirical max: 33 bytes (provides ~2x margin)
//   - Single replacement must fit: pending buffer holds exactly one
//     replacement at a time (we finish draining before matching next)
//   - 64 bytes is well under the 255-byte format maximum
//   - Larger would waste state memory with no behavioral benefit
//
// Streaming guarantee: Any replacement string produced by a trie match will
// fit entirely in the pending buffer. The caller can provide arbitrarily
// small output buffers (even 1 byte) and receive byte-identical results to
// batch mode, just requiring more calls to drain the pending buffer.
#define IREE_TOKENIZER_PRECOMPILED_MAX_PENDING 64

// Maximum grapheme size for grapheme-first trie lookup.
//
// HuggingFace tokenizers try matching whole graphemes before falling back
// to character-by-character matching. This optimization is applied only to
// graphemes smaller than this threshold.
//
// Source: tokenizers/src/normalizers/precompiled.rs
//   if grapheme.len() < 6 {
//       if let Some(norm) = self.transform(grapheme) { return; }
//   }
//   // Character-by-character fallback
//
// Why this exists:
//   - Grapheme clusters can be arbitrarily long (emoji + ZWJ sequences)
//   - Most graphemes that benefit from whole-grapheme matching are short
//   - Threshold prevents expensive failed lookups for long graphemes
//
// Streaming implication: Grapheme boundaries must be detected, but graphemes
// under 6 bytes are trivially contained within the 32-byte overlap buffer.
// No additional buffering is required for grapheme-first matching.
#define IREE_TOKENIZER_PRECOMPILED_GRAPHEME_THRESHOLD 6

// Maximum DFS stack depth for trie validation.
//
// During allocation, we validate that no trie key exceeds MAX_OVERLAP bytes.
// This requires depth-first traversal, which can have multiple pending nodes
// per level (up to 256 children at each position). We use a generous stack
// to accommodate typical tries while rejecting pathologically branching
// structures that could be crafted to cause stack overflow.
//
// 1024 entries is sufficient for:
//   - Tries with average branching factor of 32 and depth 32 (32*32=1024)
//   - All production NFKC charsmaps observed (which have low branching)
//
// Tries exceeding this are rejected as potentially malicious.
#define IREE_TOKENIZER_PRECOMPILED_VALIDATION_STACK_SIZE 1024

//===----------------------------------------------------------------------===//
// Normalizer Structure
//===----------------------------------------------------------------------===//

// Single allocation layout:
//   [iree_tokenizer_precompiled_normalizer_t]  (aligned via
//   iree_sizeof_struct) [uint32_t trie[trie_count]]                 (4-byte
//   aligned, follows struct) [uint8_t pool[pool_length]]                 (no
//   alignment needed)
typedef struct iree_tokenizer_precompiled_normalizer_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
  // Trie entry count (for bounds checking).
  iree_host_size_t trie_count;
  // Points into trailing allocation.
  const uint32_t* trie;
  // Normalized string pool length.
  iree_host_size_t pool_length;
  // Points into trailing allocation (length-prefixed format).
  const uint8_t* pool;
} iree_tokenizer_precompiled_normalizer_t;

// State for streaming precompiled normalization.
typedef struct iree_tokenizer_precompiled_state_t {
  iree_tokenizer_normalizer_state_t base;
  // Cached from normalizer (avoid pointer chase in hot path).
  iree_host_size_t trie_count;
  const uint32_t* trie;
  iree_host_size_t pool_length;
  const uint8_t* pool;
  // Overlap buffer for patterns spanning chunk boundaries.
  iree_host_size_t overlap_length;
  uint8_t overlap_buffer[IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP];
  // Pending output buffer for replacement strings.
  iree_host_size_t pending_length;
  iree_host_size_t pending_offset;  // Current drain position.
  uint8_t pending_buffer[IREE_TOKENIZER_PRECOMPILED_MAX_PENDING];
} iree_tokenizer_precompiled_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_precompiled_normalizer_vtable;

//===----------------------------------------------------------------------===//
// Double-Array Trie Helpers
//===----------------------------------------------------------------------===//

// Unit bit layout for non-leaf: offset(bits 10-30), shift_flag(bit 9),
//                               has_leaf(bit 8), label(bits 0-7)
// Unit bit layout for leaf: is_leaf(bit 31), value(bits 0-30)

static inline bool iree_tokenizer_precompiled_unit_is_leaf(uint32_t unit) {
  return (unit & 0x80000000) != 0;
}

static inline uint32_t iree_tokenizer_precompiled_unit_value(uint32_t unit) {
  return unit & 0x7FFFFFFF;
}

static inline iree_host_size_t iree_tokenizer_precompiled_unit_offset(
    uint32_t unit) {
  // offset = (unit >> 10) << (shift * 3) where shift = (unit >> 9) & 1
  return ((iree_host_size_t)(unit >> 10)) << (((unit >> 9) & 1) * 3);
}

static inline bool iree_tokenizer_precompiled_unit_has_leaf(uint32_t unit) {
  return (unit & 0x100) != 0;  // Bit 8.
}

static inline uint8_t iree_tokenizer_precompiled_unit_label(uint32_t unit) {
  return (uint8_t)(unit & 0xFF);
}

// Performs common prefix search on the double-array trie.
// Returns the offset into the normalized string pool for the longest match,
// or -1 if no match is found. |out_matched_length| receives the length of
// the matched key in bytes.
//
// Safety: The loop is bounded by key_length (input size), and bounds checks
// prevent OOB reads. Even with malicious trie data containing cycles, the
// traversal terminates after at most key_length iterations.
static int32_t iree_tokenizer_precompiled_trie_lookup(
    const iree_tokenizer_precompiled_state_t* state, const uint8_t* key,
    iree_host_size_t key_length, iree_host_size_t* out_matched_length) {
  if (state->trie_count == 0) {
    *out_matched_length = 0;
    return -1;
  }

  iree_host_size_t node_position = 0;
  uint32_t unit = state->trie[node_position];
  node_position ^= iree_tokenizer_precompiled_unit_offset(unit);

  int32_t best_value = -1;
  iree_host_size_t best_length = 0;

  // Loop bounded by key_length ensures termination even with malicious tries.
  for (iree_host_size_t i = 0; i < key_length; ++i) {
    uint8_t c = key[i];
    // Note: We handle all byte values including 0x00 (null). The key_length
    // parameter provides the boundary, not null-termination.

    node_position ^= c;
    if (node_position >= state->trie_count) break;

    unit = state->trie[node_position];
    if (iree_tokenizer_precompiled_unit_label(unit) != c) break;

    node_position ^= iree_tokenizer_precompiled_unit_offset(unit);

    if (iree_tokenizer_precompiled_unit_has_leaf(unit)) {
      if (node_position < state->trie_count) {
        uint32_t leaf_unit = state->trie[node_position];
        if (iree_tokenizer_precompiled_unit_is_leaf(leaf_unit)) {
          best_value =
              (int32_t)iree_tokenizer_precompiled_unit_value(leaf_unit);
          best_length = i + 1;
        }
      }
    }
  }

  *out_matched_length = best_length;
  return best_value;
}

// Gets the length-prefixed string at the given offset in the pool.
//
// Pool format: [LEN][content] where LEN is a single byte (0-255).
// The offset points to the content, with the length byte at (offset - 1).
//
// Preconditions (all validated at allocation time):
// - offset >= 1 (translated from original offset 0+)
// - offset - 1 contains a valid length byte (< 256)
// - offset + length <= pool_length
static iree_string_view_t iree_tokenizer_precompiled_get_replacement(
    const iree_tokenizer_precompiled_state_t* state, uint32_t offset) {
  // Length byte is immediately before content.
  uint8_t length = state->pool[offset - 1];
  const char* content = (const char*)(state->pool + offset);
  return iree_make_string_view(content, length);
}

//===----------------------------------------------------------------------===//
// Trie Validation
//===----------------------------------------------------------------------===//

// Helper: checks if offset is a valid string start in a sequential pool.
static bool iree_tokenizer_precompiled_is_valid_string_start(
    const uint8_t* pool, iree_host_size_t pool_length,
    iree_host_size_t offset) {
  if (offset == 0) return true;
  if (offset >= pool_length) return false;
  return pool[offset - 1] == '\0';
}

// DFS stack entry for trie validation traversal.
typedef struct iree_tokenizer_precompiled_validation_entry_t {
  iree_host_size_t position;
  iree_host_size_t depth;
} iree_tokenizer_precompiled_validation_entry_t;

// Validates precompiled trie structure before allocation.
static iree_status_t iree_tokenizer_precompiled_trie_validate(
    const uint32_t* trie, iree_host_size_t trie_count, const uint8_t* pool,
    iree_host_size_t pool_length) {
  // Phase 0: Validate pool size fits in 31 bits.
  if (pool_length > INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool size %" PRIhsz
                            " exceeds maximum %d for 31-bit leaf offsets",
                            pool_length, INT32_MAX);
  }

  // Phase 1: Validate all leaf pool references.
  for (iree_host_size_t i = 0; i < trie_count; ++i) {
    uint32_t unit = iree_unaligned_load_le(&trie[i]);
    if (iree_tokenizer_precompiled_unit_is_leaf(unit)) {
      uint32_t pool_offset = iree_tokenizer_precompiled_unit_value(unit);
      if (pool_offset >= pool_length) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "trie leaf at index %" PRIhsz
            " has invalid pool offset %u (pool size: %" PRIhsz ")",
            i, pool_offset, pool_length);
      }
      if (!iree_tokenizer_precompiled_is_valid_string_start(pool, pool_length,
                                                            pool_offset)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "trie leaf at index %" PRIhsz
                                " has offset %u pointing mid-string",
                                i, pool_offset);
      }
      // Verify null terminator exists within pool bounds.
      const uint8_t* string = pool + pool_offset;
      const uint8_t* end = pool + pool_length;
      while (string < end && *string != '\0') ++string;
      if (string == end) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "trie leaf at index %" PRIhsz
                                " references unterminated string at offset %u",
                                i, pool_offset);
      }
    }
  }

  // Phase 2: Verify pool strings fit in pending buffer.
  // The pending buffer (MAX_PENDING bytes) holds replacement strings when
  // output is full. Reject any replacement longer than the buffer.
  iree_host_size_t position = 0;
  while (position < pool_length) {
    const uint8_t* string_start = pool + position;
    const uint8_t* pool_end = pool + pool_length;
    const uint8_t* scan = string_start;
    while (scan < pool_end && *scan != '\0') ++scan;

    if (scan == pool_end) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pool has unterminated data at offset %" PRIhsz,
                              position);
    }

    iree_host_size_t length = (iree_host_size_t)(scan - string_start);
    if (length > IREE_TOKENIZER_PRECOMPILED_MAX_PENDING) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "replacement at offset %" PRIhsz
                              " has length "
                              "%" PRIhsz ", exceeds pending buffer (%d bytes)",
                              position, length,
                              IREE_TOKENIZER_PRECOMPILED_MAX_PENDING);
    }

    position += length + 1;
  }

  // Phase 3: Compute maximum trie key depth via depth-first traversal.
  // The overlap buffer (MAX_OVERLAP bytes) holds unprocessed input across
  // chunk boundaries. Keys longer than this cause non-deterministic streaming.
  if (trie_count > 0) {
    // Stack for iterative DFS: stores (position, depth) pairs.
    // DFS can have multiple pending children per level, so we use a generous
    // stack and reject tries that would overflow it.
    iree_tokenizer_precompiled_validation_entry_t
        stack[IREE_TOKENIZER_PRECOMPILED_VALIDATION_STACK_SIZE];
    iree_host_size_t stack_size = 0;

    // Start from root.
    uint32_t root_unit = iree_unaligned_load_le(&trie[0]);
    iree_host_size_t root_offset = ((iree_host_size_t)(root_unit >> 10))
                                   << (((root_unit >> 9) & 1) * 3);
    stack[stack_size++] =
        (iree_tokenizer_precompiled_validation_entry_t){root_offset, 0};

    while (stack_size > 0) {
      iree_host_size_t position = stack[stack_size - 1].position;
      iree_host_size_t depth = stack[stack_size - 1].depth;
      --stack_size;

      // Try all 256 possible byte values at this position.
      for (uint32_t c = 0; c < 256; ++c) {
        iree_host_size_t child_position = position ^ c;
        if (child_position >= trie_count) continue;

        uint32_t unit = iree_unaligned_load_le(&trie[child_position]);
        // Skip unallocated positions (zero units). A zero unit has label=0,
        // which would falsely match c=0 and create phantom self-loop edges.
        if (unit == 0) continue;
        uint8_t label = (uint8_t)(unit & 0xFF);
        if (label != c) continue;  // Label mismatch, not a valid edge.

        iree_host_size_t child_depth = depth + 1;

        // Reject if key depth exceeds overlap buffer.
        if (child_depth > IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "trie key exceeds %d bytes (overlap buffer limit)",
              IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP);
        }

        // Continue traversal to find longer keys.
        iree_host_size_t child_offset = ((iree_host_size_t)(unit >> 10))
                                        << (((unit >> 9) & 1) * 3);
        iree_host_size_t next_position = child_position ^ child_offset;
        if (next_position < trie_count) {
          // Reject if stack would overflow (indicates unusual trie structure).
          if (stack_size >= IREE_TOKENIZER_PRECOMPILED_VALIDATION_STACK_SIZE) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "trie structure too complex for validation");
          }
          stack[stack_size++] = (iree_tokenizer_precompiled_validation_entry_t){
              next_position, child_depth};
        }
      }
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_precompiled_normalizer_allocate(
    iree_const_byte_span_t data, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  // Parse the precompiled_charsmap format.
  if (data.data_length < 4) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "precompiled data too short: %" PRIhsz " bytes",
                             data.data_length));
  }

  uint32_t trie_size_bytes = iree_unaligned_load_le((const uint32_t*)data.data);
  if (trie_size_bytes % sizeof(uint32_t) != 0) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "precompiled trie size not aligned: %u bytes",
                             trie_size_bytes));
  }
  if (trie_size_bytes > data.data_length - 4) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "precompiled trie size invalid: %u bytes (have %" PRIhsz ")",
                trie_size_bytes, data.data_length));
  }
  iree_host_size_t trie_count = trie_size_bytes / sizeof(uint32_t);
  iree_host_size_t pool_length = data.data_length - 4 - trie_size_bytes;

  // Calculate single allocation size.
  iree_host_size_t struct_size =
      iree_sizeof_struct(iree_tokenizer_precompiled_normalizer_t);
  iree_host_size_t data_size = trie_size_bytes + pool_length;
  if (data_size > IREE_HOST_SIZE_MAX - struct_size) {
    IREE_RETURN_AND_END_ZONE(
        z0,
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "precompiled normalizer allocation size overflow"));
  }
  iree_host_size_t total_size = struct_size + data_size;

  // Validate trie structure BEFORE allocating.
  const uint32_t* input_trie = (const uint32_t*)(data.data + 4);
  const uint8_t* input_pool = data.data + 4 + trie_size_bytes;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_precompiled_trie_validate(input_trie, trie_count,
                                                   input_pool, pool_length));

  // Single allocation for normalizer + trie + pool.
  iree_tokenizer_precompiled_normalizer_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&normalizer));

  memset(normalizer, 0, sizeof(*normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_precompiled_normalizer_vtable,
      sizeof(iree_tokenizer_precompiled_state_t));
  normalizer->allocator = allocator;

  normalizer->trie_count = trie_count;
  normalizer->pool_length = pool_length;

  // Point trie into trailing allocation.
  uint32_t* trie_storage =
      (uint32_t*)((uint8_t*)normalizer +
                  iree_sizeof_struct(iree_tokenizer_precompiled_normalizer_t));
  normalizer->trie = trie_storage;

  // Copy trie, translating leaf offsets for length-prefix format.
  for (iree_host_size_t i = 0; i < trie_count; ++i) {
    uint32_t unit = iree_unaligned_load_le(&input_trie[i]);
    if (iree_tokenizer_precompiled_unit_is_leaf(unit)) {
      uint32_t old_offset = iree_tokenizer_precompiled_unit_value(unit);
      uint32_t new_offset = old_offset + 1;
      unit = 0x80000000 | new_offset;
    }
    trie_storage[i] = unit;
  }

  // Transform pool to length-prefix format.
  if (pool_length > 0) {
    uint8_t* pool_storage = (uint8_t*)(trie_storage + trie_count);
    normalizer->pool = pool_storage;

    iree_host_size_t input_pos = 0;
    iree_host_size_t output_pos = 0;
    while (input_pos < pool_length) {
      const uint8_t* string_start = input_pool + input_pos;
      const uint8_t* scan = string_start;
      while (*scan != '\0') ++scan;
      iree_host_size_t length = (iree_host_size_t)(scan - string_start);

      pool_storage[output_pos++] = (uint8_t)length;
      if (length > 0) {
        memcpy(pool_storage + output_pos, input_pool + input_pos, length);
        output_pos += length;
      }
      input_pos += length + 1;
    }
    IREE_ASSERT(output_pos == pool_length);
  }

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_precompiled_normalizer_destroy(
    iree_tokenizer_normalizer_t* base_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_precompiled_normalizer_t* normalizer =
      (iree_tokenizer_precompiled_normalizer_t*)base_normalizer;
  iree_allocator_t allocator = normalizer->allocator;
  iree_allocator_free(allocator, normalizer);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// State Management
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_precompiled_state_initialize(
    const iree_tokenizer_normalizer_t* base_normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_precompiled_normalizer_t* normalizer =
      (const iree_tokenizer_precompiled_normalizer_t*)base_normalizer;
  iree_tokenizer_precompiled_state_t* state =
      (iree_tokenizer_precompiled_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.normalizer = base_normalizer;

  // Cache hot data from normalizer.
  state->trie_count = normalizer->trie_count;
  state->trie = normalizer->trie;
  state->pool_length = normalizer->pool_length;
  state->pool = normalizer->pool;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_precompiled_state_deinitialize(
    iree_tokenizer_normalizer_state_t* base_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // No resources to release.
  (void)base_state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Output Helpers
//===----------------------------------------------------------------------===//

// Drains pending buffer to output. Returns bytes written.
static iree_host_size_t iree_tokenizer_precompiled_drain_pending(
    iree_tokenizer_precompiled_state_t* state, uint8_t* output,
    iree_host_size_t output_capacity) {
  iree_host_size_t available = state->pending_length - state->pending_offset;
  iree_host_size_t to_write =
      available < output_capacity ? available : output_capacity;
  if (to_write > 0) {
    memcpy(output, state->pending_buffer + state->pending_offset, to_write);
    state->pending_offset += to_write;
    if (state->pending_offset >= state->pending_length) {
      state->pending_length = 0;
      state->pending_offset = 0;
    }
  }
  return to_write;
}

// Emits bytes to output, using pending buffer if needed. Returns bytes written.
static iree_host_size_t iree_tokenizer_precompiled_emit(
    iree_tokenizer_precompiled_state_t* state, const uint8_t* data,
    iree_host_size_t length, uint8_t* output,
    iree_host_size_t output_capacity) {
  if (length == 0) return 0;

  if (length <= output_capacity) {
    memcpy(output, data, length);
    return length;
  }

  // Doesn't fully fit - use pending buffer.
  IREE_ASSERT(state->pending_length == 0);
  IREE_ASSERT(length <= IREE_TOKENIZER_PRECOMPILED_MAX_PENDING);
  memcpy(state->pending_buffer, data, length);
  state->pending_length = length;
  state->pending_offset = 0;
  return iree_tokenizer_precompiled_drain_pending(state, output,
                                                  output_capacity);
}

//===----------------------------------------------------------------------===//
// Grapheme Processing
//===----------------------------------------------------------------------===//

// Finds the next grapheme boundary in the input.
// Returns the length in bytes of the first grapheme.
static iree_host_size_t iree_tokenizer_precompiled_grapheme_length(
    const uint8_t* data, iree_host_size_t length) {
  if (length == 0) return 0;

  // Decode first codepoint.
  iree_string_view_t view = {(const char*)data, length};
  iree_host_size_t position = 0;
  uint32_t first_codepoint = iree_unicode_utf8_decode(view, &position);
  if (first_codepoint == 0 && position == 0) {
    // Invalid UTF-8 - treat single byte as grapheme.
    return 1;
  }

  iree_host_size_t grapheme_end = position;

  // Extend grapheme to include following combining marks.
  while (grapheme_end < length) {
    iree_string_view_t remaining = {(const char*)(data + grapheme_end),
                                    length - grapheme_end};
    iree_host_size_t rel_pos = 0;
    uint32_t codepoint = iree_unicode_utf8_decode(remaining, &rel_pos);
    if (codepoint == 0 && rel_pos == 0) break;

    // Check if this is a combining mark (CCC > 0).
    uint8_t ccc = iree_unicode_ccc(codepoint);
    if (ccc == 0) break;  // Not a combining mark - end of grapheme.

    grapheme_end += rel_pos;
  }

  return grapheme_end;
}

// Processes a single grapheme using grapheme-first matching.
// Returns the number of input bytes consumed.
static iree_host_size_t iree_tokenizer_precompiled_process_grapheme(
    iree_tokenizer_precompiled_state_t* state, const uint8_t* data,
    iree_host_size_t grapheme_length, iree_host_size_t total_available,
    uint8_t** out_ptr, uint8_t* out_end) {
  iree_host_size_t consumed = 0;

  // Step 1: Try whole-grapheme match if grapheme is short.
  if (grapheme_length < IREE_TOKENIZER_PRECOMPILED_GRAPHEME_THRESHOLD) {
    iree_host_size_t matched_length = 0;
    int32_t norm_offset = iree_tokenizer_precompiled_trie_lookup(
        state, data, grapheme_length, &matched_length);
    if (norm_offset >= 0 && matched_length == grapheme_length) {
      // Whole grapheme matched - emit replacement.
      iree_string_view_t replacement =
          iree_tokenizer_precompiled_get_replacement(state,
                                                     (uint32_t)norm_offset);
      iree_host_size_t remaining = (iree_host_size_t)(out_end - *out_ptr);
      iree_host_size_t written = iree_tokenizer_precompiled_emit(
          state, (const uint8_t*)replacement.data, replacement.size, *out_ptr,
          remaining);
      *out_ptr += written;
      return grapheme_length;
    }
  }

  // Step 2: Character-by-character fallback.
  while (consumed < grapheme_length) {
    // Check if we have pending output to drain.
    if (state->pending_length > state->pending_offset) {
      break;  // Caller must drain pending first.
    }

    iree_host_size_t remaining_input = total_available - consumed;
    iree_host_size_t matched_length = 0;
    int32_t norm_offset = iree_tokenizer_precompiled_trie_lookup(
        state, data + consumed, remaining_input, &matched_length);

    iree_host_size_t remaining_output = (iree_host_size_t)(out_end - *out_ptr);

    if (norm_offset >= 0 && matched_length > 0) {
      // Match found - emit replacement.
      iree_string_view_t replacement =
          iree_tokenizer_precompiled_get_replacement(state,
                                                     (uint32_t)norm_offset);
      iree_host_size_t written = iree_tokenizer_precompiled_emit(
          state, (const uint8_t*)replacement.data, replacement.size, *out_ptr,
          remaining_output);
      *out_ptr += written;
      consumed += matched_length;
    } else {
      // No match - emit one UTF-8 character as-is.
      uint8_t first_byte = data[consumed];
      iree_host_size_t char_length =
          iree_unicode_utf8_sequence_length(first_byte);
      if (char_length == 0) char_length = 1;  // Invalid byte.
      if (consumed + char_length > grapheme_length) {
        char_length = grapheme_length - consumed;
      }

      iree_host_size_t written = iree_tokenizer_precompiled_emit(
          state, data + consumed, char_length, *out_ptr, remaining_output);
      *out_ptr += written;
      consumed += char_length;
    }

    // If emit used pending buffer, stop processing.
    if (state->pending_length > state->pending_offset) {
      break;
    }
  }

  return consumed;
}

//===----------------------------------------------------------------------===//
// Normalizer Vtable Implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_precompiled_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  (void)flags;  // Precompiled handles chunk boundaries via overlap buffer.

  iree_tokenizer_precompiled_state_t* state =
      (iree_tokenizer_precompiled_state_t*)base_state;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;
  iree_host_size_t input_consumed = 0;

  // Drain any pending output from previous call.
  if (state->pending_length > state->pending_offset) {
    iree_host_size_t drained = iree_tokenizer_precompiled_drain_pending(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += drained;
    if (state->pending_length > state->pending_offset) {
      *out_consumed = 0;
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Empty trie: passthrough.
  if (state->trie_count == 0) {
    iree_host_size_t to_copy =
        input.size < output.size ? input.size : output.size;
    if (to_copy > 0) {
      memcpy(out_ptr, input.data, to_copy);
      out_ptr += to_copy;
      input_consumed = to_copy;
    }
    *out_consumed = input_consumed;
    *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
    return iree_ok_status();
  }

  // Build working buffer from overlap + input.
  uint8_t work_buffer[2 * IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP];
  iree_host_size_t work_length = 0;

  // Save old overlap length before it gets overwritten - needed to compute
  // how much new input was consumed vs saved.
  iree_host_size_t old_overlap_length = state->overlap_length;

  if (old_overlap_length > 0) {
    memcpy(work_buffer, state->overlap_buffer, old_overlap_length);
    work_length = old_overlap_length;
  }

  iree_host_size_t input_to_process =
      input.size < IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP
          ? input.size
          : IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP;
  if (input_to_process > 0) {
    memcpy(work_buffer + work_length, input.data, input_to_process);
    work_length += input_to_process;
  }

  // Calculate safe processing region.
  iree_host_size_t safe_length =
      work_length > IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP
          ? work_length - IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP
          : 0;

  iree_host_size_t work_position = 0;

  // Process graphemes within safe region.
  while (work_position < safe_length && out_ptr < out_end) {
    if (state->pending_length > state->pending_offset) break;

    iree_host_size_t grapheme_length =
        iree_tokenizer_precompiled_grapheme_length(work_buffer + work_position,
                                                   work_length - work_position);
    if (grapheme_length == 0) break;

    iree_host_size_t consumed = iree_tokenizer_precompiled_process_grapheme(
        state, work_buffer + work_position, grapheme_length,
        work_length - work_position, &out_ptr, out_end);
    work_position += consumed;

    if (consumed < grapheme_length) break;  // Incomplete processing.
  }

  // Save unprocessed tail to overlap buffer.
  // Key invariant: every input byte is either processed OR saved. We only
  // claim input as consumed if it meets one of these conditions.
  iree_host_size_t tail_length = work_length - work_position;

  // Track how much of the tail comes from old overlap vs new input.
  iree_host_size_t old_overlap_remaining =
      work_position < old_overlap_length ? old_overlap_length - work_position
                                         : 0;
  iree_host_size_t input_in_tail = tail_length - old_overlap_remaining;

  // How many bytes of new input were fully processed (contributed to output)?
  iree_host_size_t input_processed = work_position > old_overlap_length
                                         ? work_position - old_overlap_length
                                         : 0;

  // Compute how much input we can save in the overlap buffer.
  // Old overlap bytes have priority (they were already "consumed" previously).
  iree_host_size_t input_saved;
  if (tail_length <= IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP) {
    // Tail fits entirely - save all of it.
    input_saved = input_in_tail;
    if (tail_length > 0) {
      memcpy(state->overlap_buffer, work_buffer + work_position, tail_length);
    }
    state->overlap_length = tail_length;
  } else {
    // Tail exceeds buffer capacity. Save old_overlap_remaining (which must
    // fit since it was <= MAX_OVERLAP) plus as much input as space allows.
    iree_host_size_t space_for_input =
        IREE_TOKENIZER_PRECOMPILED_MAX_OVERLAP - old_overlap_remaining;
    input_saved =
        input_in_tail < space_for_input ? input_in_tail : space_for_input;

    iree_host_size_t new_overlap_length = old_overlap_remaining + input_saved;
    memcpy(state->overlap_buffer, work_buffer + work_position,
           new_overlap_length);
    state->overlap_length = new_overlap_length;
  }

  // Only claim input bytes that were processed (output written) OR saved
  // (buffered for later). Any input bytes beyond this will be resent by
  // the caller on the next call.
  input_consumed = input_processed + input_saved;

  *out_consumed = input_consumed;
  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_precompiled_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_precompiled_state_t* state =
      (iree_tokenizer_precompiled_state_t*)base_state;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;

  // Drain pending output.
  if (state->pending_length > state->pending_offset) {
    iree_host_size_t drained = iree_tokenizer_precompiled_drain_pending(
        state, out_ptr, (iree_host_size_t)(out_end - out_ptr));
    out_ptr += drained;
    if (state->pending_length > state->pending_offset) {
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
  }

  // Process remaining overlap buffer as final data.
  while (state->overlap_length > 0 && out_ptr < out_end) {
    if (state->pending_length > state->pending_offset) break;

    iree_host_size_t grapheme_length =
        iree_tokenizer_precompiled_grapheme_length(state->overlap_buffer,
                                                   state->overlap_length);
    if (grapheme_length == 0) {
      // Emit remaining bytes as-is.
      iree_host_size_t remaining = (iree_host_size_t)(out_end - out_ptr);
      iree_host_size_t to_emit =
          state->overlap_length < remaining ? state->overlap_length : remaining;
      memcpy(out_ptr, state->overlap_buffer, to_emit);
      out_ptr += to_emit;
      if (to_emit < state->overlap_length) {
        memmove(state->overlap_buffer, state->overlap_buffer + to_emit,
                state->overlap_length - to_emit);
        state->overlap_length -= to_emit;
      } else {
        state->overlap_length = 0;
      }
      break;
    }

    iree_host_size_t consumed = iree_tokenizer_precompiled_process_grapheme(
        state, state->overlap_buffer, grapheme_length, state->overlap_length,
        &out_ptr, out_end);

    if (consumed > 0 && consumed <= state->overlap_length) {
      memmove(state->overlap_buffer, state->overlap_buffer + consumed,
              state->overlap_length - consumed);
      state->overlap_length -= consumed;
    }

    if (consumed < grapheme_length) break;
  }

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static bool iree_tokenizer_precompiled_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_precompiled_state_t* state =
      (const iree_tokenizer_precompiled_state_t*)base_state;
  // Report pending if we have:
  // 1. OUTPUT waiting in pending_buffer (replacement strings pending emission)
  // 2. INPUT buffered in overlap_buffer (bytes awaiting grapheme completion)
  //
  // The overlap buffer check is critical for special token ordering: when text
  // precedes a special token but is still being normalized, we must defer the
  // special token emission until the buffered text is fully processed. Without
  // this check, special tokens can incorrectly be emitted before preceding text
  // when the normalizer holds content in its overlap buffer.
  return state->pending_length > state->pending_offset ||
         state->overlap_length > 0;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_precompiled_normalizer_vtable = {
        .destroy = iree_tokenizer_precompiled_normalizer_destroy,
        .state_initialize = iree_tokenizer_precompiled_state_initialize,
        .state_deinitialize = iree_tokenizer_precompiled_state_deinitialize,
        .state_process = iree_tokenizer_precompiled_state_process,
        .state_finalize = iree_tokenizer_precompiled_state_finalize,
        .state_has_pending = iree_tokenizer_precompiled_state_has_pending,
};
